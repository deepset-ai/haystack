import logging
from math import isclose

import numpy as np
import pandas as pd
from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.memory import InMemoryDocumentStore
import pytest
from elasticsearch import Elasticsearch

from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores import MilvusDocumentStore
from haystack.nodes.retriever.dense import DensePassageRetriever, EmbeddingRetriever, TableTextRetriever
from haystack.nodes.retriever.sparse import BM25Retriever, FilterRetriever, TfidfRetriever
from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

from ..conftest import SAMPLES_PATH, MockFilterRetriever


# TODO check if we this works with only "memory" arg
@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("mdr", "elasticsearch"),
        ("mdr", "faiss"),
        ("mdr", "memory"),
        ("mdr", "milvus1"),
        ("dpr", "elasticsearch"),
        ("dpr", "faiss"),
        ("dpr", "memory"),
        ("dpr", "milvus1"),
        ("embedding", "elasticsearch"),
        ("embedding", "faiss"),
        ("embedding", "memory"),
        ("embedding", "milvus1"),
        ("elasticsearch", "elasticsearch"),
        ("es_filter_only", "elasticsearch"),
        ("tfidf", "memory"),
    ],
    indirect=True,
)
def test_retrieval(retriever_with_docs: BaseRetriever, document_store_with_docs: BaseDocumentStore):
    if not isinstance(retriever_with_docs, (BM25Retriever, FilterRetriever, TfidfRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    # test without filters
    # NOTE: FilterRetriever simply returns all documents matching a filter,
    # so without filters applied it does nothing
    if not isinstance(retriever_with_docs, FilterRetriever):
        res = retriever_with_docs.retrieve(query="Who lives in Berlin?")
        assert res[0].content == "My name is Carla and I live in Berlin"
        assert len(res) == 5
        assert res[0].meta["name"] == "filename1"

    # test with filters
    if not isinstance(document_store_with_docs, (FAISSDocumentStore, MilvusDocumentStore)) and not isinstance(
        retriever_with_docs, TfidfRetriever
    ):
        # single filter
        result = retriever_with_docs.retrieve(query="Christelle", filters={"name": ["filename3"]}, top_k=5)
        assert len(result) == 1
        assert type(result[0]) == Document
        assert result[0].content == "My name is Christelle and I live in Paris"
        assert result[0].meta["name"] == "filename3"

        # multiple filters
        result = retriever_with_docs.retrieve(
            query="Paul", filters={"name": ["filename2"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 1
        assert type(result[0]) == Document
        assert result[0].meta["name"] == "filename2"

        result = retriever_with_docs.retrieve(
            query="Carla", filters={"name": ["filename1"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 0

    # test FilterRetriever with empty query using the run() method
    if isinstance(retriever_with_docs, FilterRetriever):
        result = retriever_with_docs.run(root_node="Query", query="", filters={"name": ["filename2"]})
        assert len(result) == 2
        result = result[0]
        assert "documents" in result
        result = result["documents"]
        assert len(result) == 1
        assert type(result[0]) == Document
        assert result[0].content == "My name is Paul and I live in New York"
        assert result[0].meta["name"] == "filename2"


def test_retrieval_empty_query(document_store_with_docs: BaseDocumentStore):
    # test with empty query using the run() method
    mock_document = Document(id="0", content="test")
    retriever = MockFilterRetriever(document_store=document_store_with_docs, mock_document=mock_document)
    result = retriever.run(root_node="Query", query="", filters={})
    assert result[0]["documents"][0] == mock_document


def test_batch_retrieval_single_query(retriever_with_docs, document_store_with_docs):
    if not isinstance(retriever_with_docs, (BM25Retriever, FilterRetriever, TfidfRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    res = retriever_with_docs.retrieve_batch(queries=["Who lives in Berlin?"])

    # Expected return type: List of lists of Documents
    assert isinstance(res, list)
    assert isinstance(res[0], list)
    assert isinstance(res[0][0], Document)

    assert len(res) == 1
    assert len(res[0]) == 5
    assert res[0][0].content == "My name is Carla and I live in Berlin"
    assert res[0][0].meta["name"] == "filename1"


def test_batch_retrieval_multiple_queries(retriever_with_docs, document_store_with_docs):
    if not isinstance(retriever_with_docs, (BM25Retriever, FilterRetriever, TfidfRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    res = retriever_with_docs.retrieve_batch(queries=["Who lives in Berlin?", "Who lives in New York?"])

    # Expected return type: list of lists of Documents
    assert isinstance(res, list)
    assert isinstance(res[0], list)
    assert isinstance(res[0][0], Document)

    assert res[0][0].content == "My name is Carla and I live in Berlin"
    assert len(res[0]) == 5
    assert res[0][0].meta["name"] == "filename1"

    assert res[1][0].content == "My name is Paul and I live in New York"
    assert len(res[1]) == 5
    assert res[1][0].meta["name"] == "filename2"


@pytest.mark.elasticsearch
def test_elasticsearch_custom_query():
    client = Elasticsearch()
    client.indices.delete(index="haystack_test_custom", ignore=[404])
    document_store = ElasticsearchDocumentStore(
        index="haystack_test_custom", content_field="custom_text_field", embedding_field="custom_embedding_field"
    )
    documents = [
        {"content": "test_1", "meta": {"year": "2019"}},
        {"content": "test_2", "meta": {"year": "2020"}},
        {"content": "test_3", "meta": {"year": "2021"}},
        {"content": "test_4", "meta": {"year": "2021"}},
        {"content": "test_5", "meta": {"year": "2021"}},
    ]
    document_store.write_documents(documents)

    # test custom "terms" query
    retriever = BM25Retriever(
        document_store=document_store,
        custom_query="""
            {
                "size": 10,
                "query": {
                    "bool": {
                        "should": [{
                            "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["content"]}}],
                            "filter": [{"terms": {"year": ${years}}}]}}}""",
    )
    results = retriever.retrieve(query="test", filters={"years": ["2020", "2021"]})
    assert len(results) == 4

    # test custom "term" query
    retriever = BM25Retriever(
        document_store=document_store,
        custom_query="""
                {
                    "size": 10,
                    "query": {
                        "bool": {
                            "should": [{
                                "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["content"]}}],
                                "filter": [{"term": {"year": ${years}}}]}}}""",
    )
    results = retriever.retrieve(query="test", filters={"years": "2021"})
    assert len(results) == 3


@pytest.mark.integration
@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"], indirect=True
)
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
def test_dpr_embedding(document_store: BaseDocumentStore, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs.sort(key=lambda d: d.id)

    print([doc.id for doc in docs])

    expected_values = [0.00892, 0.00780, 0.00482, -0.00626, 0.010966]
    for doc, expected_value in zip(docs, expected_values):
        embedding = doc.embedding
        # always normalize vector as faiss returns normalized vectors and other document stores do not
        embedding /= np.linalg.norm(embedding)
        assert len(embedding) == 768
        assert isclose(embedding[0], expected_value, rel_tol=0.01)


@pytest.mark.integration
@pytest.mark.parametrize(
    "document_store", ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"], indirect=True
)
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.embedding_dim(128)
def test_retribert_embedding(document_store, retriever, docs_with_ids):
    if isinstance(document_store, WeaviateDocumentStore):
        # Weaviate sets the embedding dimension to 768 as soon as it is initialized.
        # We need 128 here and therefore initialize a new WeaviateDocumentStore.
        document_store = WeaviateDocumentStore(index="haystack_test", embedding_dim=128, recreate_index=True)
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs = sorted(docs, key=lambda d: d.id)

    expected_values = [0.14017, 0.05975, 0.14267, 0.15099, 0.14383]
    for doc, expected_value in zip(docs, expected_values):
        embedding = doc.embedding
        assert len(embedding) == 128
        # always normalize vector as faiss returns normalized vectors and other document stores do not
        embedding /= np.linalg.norm(embedding)
        assert isclose(embedding[0], expected_value, rel_tol=0.001)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.parametrize("document_store", ["elasticsearch", "memory"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_text_retriever_embedding(document_store, retriever, docs):

    document_store.return_embedding = True
    document_store.write_documents(docs)
    table_data = {
        "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
        "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
    }
    table = pd.DataFrame(table_data)
    table_doc = Document(content=table, content_type="table", id="6")
    document_store.write_documents([table_doc])
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs = sorted(docs, key=lambda d: d.id)

    expected_values = [0.061191384, 0.038075786, 0.27447605, 0.09399721, 0.0959682]
    for doc, expected_value in zip(docs, expected_values):
        assert len(doc.embedding) == 512
        assert isclose(doc.embedding[0], expected_value, rel_tol=0.001)


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_dpr_saving_and_loading(tmp_path, retriever, document_store):
    retriever.save(f"{tmp_path}/test_dpr_save")

    def sum_params(model):
        s = []
        for p in model.parameters():
            n = p.cpu().data.numpy()
            s.append(np.sum(n))
        return sum(s)

    original_sum_query = sum_params(retriever.query_encoder)
    original_sum_passage = sum_params(retriever.passage_encoder)
    del retriever

    loaded_retriever = DensePassageRetriever.load(f"{tmp_path}/test_dpr_save", document_store)

    loaded_sum_query = sum_params(loaded_retriever.query_encoder)
    loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)

    assert abs(original_sum_query - loaded_sum_query) < 0.1
    assert abs(original_sum_passage - loaded_sum_passage) < 0.1

    # comparison of weights (RAM intense!)
    # for p1, p2 in zip(retriever.query_encoder.parameters(), loaded_retriever.query_encoder.parameters()):
    #     assert (p1.data.ne(p2.data).sum() == 0)
    #
    # for p1, p2 in zip(retriever.passage_encoder.parameters(), loaded_retriever.passage_encoder.parameters()):
    #     assert (p1.data.ne(p2.data).sum() == 0)

    # attributes
    assert loaded_retriever.processor.embed_title == True
    assert loaded_retriever.batch_size == 16
    assert loaded_retriever.processor.max_seq_len_passage == 256
    assert loaded_retriever.processor.max_seq_len_query == 64

    # Tokenizer
    assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
    assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
    assert loaded_retriever.passage_tokenizer.do_lower_case == True
    assert loaded_retriever.query_tokenizer.do_lower_case == True
    assert loaded_retriever.passage_tokenizer.vocab_size == 30522
    assert loaded_retriever.query_tokenizer.vocab_size == 30522


@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_text_retriever_saving_and_loading(tmp_path, retriever, document_store):
    retriever.save(f"{tmp_path}/test_table_text_retriever_save")

    def sum_params(model):
        s = []
        for p in model.parameters():
            n = p.cpu().data.numpy()
            s.append(np.sum(n))
        return sum(s)

    original_sum_query = sum_params(retriever.query_encoder)
    original_sum_passage = sum_params(retriever.passage_encoder)
    original_sum_table = sum_params(retriever.table_encoder)
    del retriever

    loaded_retriever = TableTextRetriever.load(f"{tmp_path}/test_table_text_retriever_save", document_store)

    loaded_sum_query = sum_params(loaded_retriever.query_encoder)
    loaded_sum_passage = sum_params(loaded_retriever.passage_encoder)
    loaded_sum_table = sum_params(loaded_retriever.table_encoder)

    assert abs(original_sum_query - loaded_sum_query) < 0.1
    assert abs(original_sum_passage - loaded_sum_passage) < 0.1
    assert abs(original_sum_table - loaded_sum_table) < 0.01

    # attributes
    assert loaded_retriever.processor.embed_meta_fields == ["name", "section_title", "caption"]
    assert loaded_retriever.batch_size == 16
    assert loaded_retriever.processor.max_seq_len_passage == 256
    assert loaded_retriever.processor.max_seq_len_table == 256
    assert loaded_retriever.processor.max_seq_len_query == 64

    # Tokenizer
    assert isinstance(loaded_retriever.passage_tokenizer, DPRContextEncoderTokenizerFast)
    assert isinstance(loaded_retriever.table_tokenizer, DPRContextEncoderTokenizerFast)
    assert isinstance(loaded_retriever.query_tokenizer, DPRQuestionEncoderTokenizerFast)
    assert loaded_retriever.passage_tokenizer.do_lower_case == True
    assert loaded_retriever.table_tokenizer.do_lower_case == True
    assert loaded_retriever.query_tokenizer.do_lower_case == True
    assert loaded_retriever.passage_tokenizer.vocab_size == 30522
    assert loaded_retriever.table_tokenizer.vocab_size == 30522
    assert loaded_retriever.query_tokenizer.vocab_size == 30522


@pytest.mark.embedding_dim(128)
def test_table_text_retriever_training(document_store):
    retriever = TableTextRetriever(
        document_store=document_store,
        query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
        passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
        table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
        use_gpu=False,
    )

    retriever.train(
        data_dir=SAMPLES_PATH / "mmr",
        train_filename="sample.json",
        n_epochs=1,
        n_gpu=0,
        save_dir="test_table_text_retriever_train",
    )

    # Load trained model
    retriever = TableTextRetriever.load(load_dir="test_table_text_retriever_train", document_store=document_store)


@pytest.mark.elasticsearch
def test_elasticsearch_highlight():
    client = Elasticsearch()
    client.indices.delete(index="haystack_hl_test", ignore=[404])

    # Mapping the content and title field as "text" perform search on these both fields.
    document_store = ElasticsearchDocumentStore(
        index="haystack_hl_test",
        content_field="title",
        custom_mapping={"mappings": {"properties": {"content": {"type": "text"}, "title": {"type": "text"}}}},
    )
    documents = [
        {
            "title": "Green tea components",
            "meta": {
                "content": "The green tea plant contains a range of healthy compounds that make it into the final drink"
            },
            "id": "1",
        },
        {
            "title": "Green tea catechin",
            "meta": {"content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG)."},
            "id": "2",
        },
        {
            "title": "Minerals in Green tea",
            "meta": {"content": "Green tea also has small amounts of minerals that can benefit your health."},
            "id": "3",
        },
        {
            "title": "Green tea Benefits",
            "meta": {"content": "Green tea does more than just keep you alert, it may also help boost brain function."},
            "id": "4",
        },
    ]
    document_store.write_documents(documents)

    # Enabled highlighting on "title"&"content" field only using custom query
    retriever_1 = BM25Retriever(
        document_store=document_store,
        custom_query="""{
            "size": 20,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": ${query},
                                "fields": [
                                    "content^3",
                                    "title^5"
                                ]
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "pre_tags": [
                    "**"
                ],
                "post_tags": [
                    "**"
                ],
                "number_of_fragments": 3,
                "fragment_size": 5,
                "fields": {
                    "content": {},
                    "title": {}
                }
            }
        }""",
    )
    results = retriever_1.retrieve(query="is green tea healthy")

    assert len(results[0].meta["highlighted"]) == 2
    assert results[0].meta["highlighted"]["title"] == ["**Green**", "**tea** components"]
    assert results[0].meta["highlighted"]["content"] == ["The **green**", "**tea** plant", "range of **healthy**"]

    # Enabled highlighting on "title" field only using custom query
    retriever_2 = BM25Retriever(
        document_store=document_store,
        custom_query="""{
            "size": 20,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": ${query},
                                "fields": [
                                    "content^3",
                                    "title^5"
                                ]
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "pre_tags": [
                    "**"
                ],
                "post_tags": [
                    "**"
                ],
                "number_of_fragments": 3,
                "fragment_size": 5,
                "fields": {
                    "title": {}
                }
            }
        }""",
    )
    results = retriever_2.retrieve(query="is green tea healthy")

    assert len(results[0].meta["highlighted"]) == 1
    assert results[0].meta["highlighted"]["title"] == ["**Green**", "**tea** components"]


def test_elasticsearch_filter_must_not_increase_results():
    index = "filter_must_not_increase_results"
    client = Elasticsearch()
    client.indices.delete(index=index, ignore=[404])
    documents = [
        {
            "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
            "meta": {"content_type": "text"},
            "id": "1",
        },
        {
            "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
            "meta": {"content_type": "text"},
            "id": "2",
        },
        {
            "content": "Green tea also has small amounts of minerals that can benefit your health.",
            "meta": {"content_type": "text"},
            "id": "3",
        },
        {
            "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
            "meta": {"content_type": "text"},
            "id": "4",
        },
    ]
    doc_store = ElasticsearchDocumentStore(index=index)
    doc_store.write_documents(documents)
    results_wo_filter = doc_store.query(query="drink")
    assert len(results_wo_filter) == 1
    results_w_filter = doc_store.query(query="drink", filters={"content_type": "text"})
    assert len(results_w_filter) == 1
    doc_store.delete_index(index)


def test_elasticsearch_all_terms_must_match():
    index = "all_terms_must_match"
    client = Elasticsearch()
    client.indices.delete(index=index, ignore=[404])
    documents = [
        {
            "content": "The green tea plant contains a range of healthy compounds that make it into the final drink",
            "meta": {"content_type": "text"},
            "id": "1",
        },
        {
            "content": "Green tea contains a catechin called epigallocatechin-3-gallate (EGCG).",
            "meta": {"content_type": "text"},
            "id": "2",
        },
        {
            "content": "Green tea also has small amounts of minerals that can benefit your health.",
            "meta": {"content_type": "text"},
            "id": "3",
        },
        {
            "content": "Green tea does more than just keep you alert, it may also help boost brain function.",
            "meta": {"content_type": "text"},
            "id": "4",
        },
    ]
    doc_store = ElasticsearchDocumentStore(index=index)
    doc_store.write_documents(documents)
    results_wo_all_terms_must_match = doc_store.query(query="drink green tea")
    assert len(results_wo_all_terms_must_match) == 4
    results_w_all_terms_must_match = doc_store.query(query="drink green tea", all_terms_must_match=True)
    assert len(results_w_all_terms_must_match) == 1
    doc_store.delete_index(index)


def test_embeddings_encoder_of_embedding_retriever_should_warn_about_model_format(caplog):
    document_store = InMemoryDocumentStore()

    with caplog.at_level(logging.WARNING):
        EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_format="farm",
        )

        assert (
            "You may need to set model_format='sentence_transformers' to ensure correct loading of model."
            in caplog.text
        )


@pytest.mark.parametrize("retriever", ["es_filter_only"], indirect=True)
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_es_filter_only(document_store, retriever):
    docs = [
        Document(content="Doc1", meta={"f1": "0"}),
        Document(content="Doc2", meta={"f1": "0"}),
        Document(content="Doc3", meta={"f1": "0"}),
        Document(content="Doc4", meta={"f1": "0"}),
        Document(content="Doc5", meta={"f1": "0"}),
        Document(content="Doc6", meta={"f1": "0"}),
        Document(content="Doc7", meta={"f1": "1"}),
        Document(content="Doc8", meta={"f1": "0"}),
        Document(content="Doc9", meta={"f1": "0"}),
        Document(content="Doc10", meta={"f1": "0"}),
        Document(content="Doc11", meta={"f1": "0"}),
        Document(content="Doc12", meta={"f1": "0"}),
    ]
    document_store.write_documents(docs)
    retrieved_docs = retriever.retrieve(query="", filters={"f1": ["0"]})
    assert len(retrieved_docs) == 11
