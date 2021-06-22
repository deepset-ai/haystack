import time

import numpy as np
import pytest
from elasticsearch import Elasticsearch
from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.milvus import MilvusDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever, ElasticsearchFilterOnlyRetriever, TfidfRetriever
from transformers import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast

DOCS = [
    Document(
        text="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
        meta={"name": "0"},
        id="1",
    ),
    Document(
        text="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
        id="2",
    ),
    Document(
        text="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
        meta={"name": "1"},
        id="3",
    ),
    Document(
        text="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
        meta={"name": "2"},
        id="4",
    ),
    Document(
        text="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
        meta={},
        id="5",
    ),
]

@pytest.mark.elasticsearch
@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("dpr", "elasticsearch"),
        ("dpr", "faiss"),
        ("dpr", "memory"),
        ("dpr", "milvus"),
        ("embedding", "elasticsearch"),
        ("embedding", "faiss"),
        ("embedding", "memory"),
        ("embedding", "milvus"),
        ("elasticsearch", "elasticsearch"),
        ("es_filter_only", "elasticsearch"),
        ("tfidf", "memory"),
    ],
    indirect=True,
)
def test_retrieval(retriever_with_docs, document_store_with_docs):
    if not isinstance(retriever_with_docs, (ElasticsearchRetriever, ElasticsearchFilterOnlyRetriever, TfidfRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    # test without filters
    res = retriever_with_docs.retrieve(query="Who lives in Berlin?")
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 3
    assert res[0].meta["name"] == "filename1"

    # test with filters
    if not isinstance(document_store_with_docs, (FAISSDocumentStore, MilvusDocumentStore)) and not isinstance(
        retriever_with_docs, TfidfRetriever
    ):
        # single filter
        result = retriever_with_docs.retrieve(query="godzilla", filters={"name": ["filename3"]}, top_k=5)
        assert len(result) == 1
        assert type(result[0]) == Document
        assert result[0].text == "My name is Christelle and I live in Paris"
        assert result[0].meta["name"] == "filename3"

        # multiple filters
        result = retriever_with_docs.retrieve(
            query="godzilla", filters={"name": ["filename2"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 1
        assert type(result[0]) == Document
        assert result[0].meta["name"] == "filename2"

        result = retriever_with_docs.retrieve(
            query="godzilla", filters={"name": ["filename1"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 0


@pytest.mark.elasticsearch
def test_elasticsearch_custom_query(elasticsearch_fixture):
    client = Elasticsearch()
    client.indices.delete(index="haystack_test_custom", ignore=[404])
    document_store = ElasticsearchDocumentStore(
        index="haystack_test_custom", text_field="custom_text_field", embedding_field="custom_embedding_field"
    )
    documents = [
        {"text": "test_1", "meta": {"year": "2019"}},
        {"text": "test_2", "meta": {"year": "2020"}},
        {"text": "test_3", "meta": {"year": "2021"}},
        {"text": "test_4", "meta": {"year": "2021"}},
        {"text": "test_5", "meta": {"year": "2021"}},
    ]
    document_store.write_documents(documents)

    # test custom "terms" query
    retriever = ElasticsearchRetriever(
        document_store=document_store,
        custom_query="""
            {
                "size": 10, 
                "query": {
                    "bool": {
                        "should": [{
                            "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["text"]}}],
                            "filter": [{"terms": {"year": ${years}}}]}}}""",
    )
    results = retriever.retrieve(query="test", filters={"years": ["2020", "2021"]})
    assert len(results) == 4

    # test custom "term" query
    retriever = ElasticsearchRetriever(
        document_store=document_store,
        custom_query="""
                {
                    "size": 10, 
                    "query": {
                        "bool": {
                            "should": [{
                                "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["text"]}}],
                                "filter": [{"term": {"year": ${years}}}]}}}""",
    )
    results = retriever.retrieve(query="test", filters={"years": "2021"})
    assert len(results) == 3


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
def test_dpr_embedding(document_store, retriever):

    document_store.return_embedding = True
    document_store.write_documents(DOCS)
    document_store.update_embeddings(retriever=retriever)
    time.sleep(1)

    doc_1 = document_store.get_document_by_id("1")
    assert len(doc_1.embedding) == 768
    assert abs(doc_1.embedding[0] - (-0.3063)) < 0.001
    doc_2 = document_store.get_document_by_id("2")
    assert abs(doc_2.embedding[0] - (-0.3914)) < 0.001
    doc_3 = document_store.get_document_by_id("3")
    assert abs(doc_3.embedding[0] - (-0.2470)) < 0.001
    doc_4 = document_store.get_document_by_id("4")
    assert abs(doc_4.embedding[0] - (-0.0802)) < 0.001
    doc_5 = document_store.get_document_by_id("5")
    assert abs(doc_5.embedding[0] - (-0.0551)) < 0.001


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
@pytest.mark.vector_dim(128)
def test_retribert_embedding(document_store, retriever):

    document_store.return_embedding = True
    document_store.write_documents(DOCS)
    document_store.update_embeddings(retriever=retriever)
    time.sleep(1)

    assert len(document_store.get_document_by_id("1").embedding) == 128
    assert abs(document_store.get_document_by_id("1").embedding[0]) < 0.6
    assert abs(document_store.get_document_by_id("2").embedding[0]) < 0.03
    assert abs(document_store.get_document_by_id("3").embedding[0]) < 0.095
    assert abs(document_store.get_document_by_id("4").embedding[0]) < 0.3
    assert abs(document_store.get_document_by_id("5").embedding[0]) < 0.32


@pytest.mark.parametrize("retriever", ["dpr"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_dpr_saving_and_loading(retriever, document_store):
    retriever.save("test_dpr_save")

    def sum_params(model):
        s = []
        for p in model.parameters():
            n = p.cpu().data.numpy()
            s.append(np.sum(n))
        return sum(s)

    original_sum_query = sum_params(retriever.query_encoder)
    original_sum_passage = sum_params(retriever.passage_encoder)
    del retriever

    loaded_retriever = DensePassageRetriever.load("test_dpr_save", document_store)

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
    assert loaded_retriever.passage_tokenizer.model_max_length == 512
    assert loaded_retriever.query_tokenizer.model_max_length == 512
