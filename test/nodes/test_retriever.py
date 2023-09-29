import logging
import os
from math import isclose
from typing import Dict, List, Optional, Union, Tuple
from unittest.mock import patch, Mock, DEFAULT

import pytest
import numpy as np
import pandas as pd
import requests
from boilerpy3.extractors import ArticleExtractor
from pandas.testing import assert_frame_equal
from transformers import PreTrainedTokenizerFast


try:
    from elasticsearch import Elasticsearch
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "elasticsearch", ie)


from haystack.document_stores.base import BaseDocumentStore, FilterType, KeywordDocumentStore
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.web import WebRetriever
from haystack.nodes.search_engine import WebSearch
from haystack.pipelines import DocumentSearchPipeline
from haystack.schema import Document
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.retriever.dense import DensePassageRetriever, EmbeddingRetriever, TableTextRetriever
from haystack.nodes.retriever.sparse import BM25Retriever, FilterRetriever, TfidfRetriever
from haystack.nodes.retriever.multimodal import MultiModalRetriever
from haystack.nodes.retriever._openai_encoder import _OpenAIEmbeddingEncoder

from ..conftest import MockBaseRetriever, fail_at_version


# TODO check if we this works with only "memory" arg
@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("mdr", "elasticsearch"),
        ("mdr", "faiss"),
        ("mdr", "memory"),
        ("dpr", "elasticsearch"),
        ("dpr", "faiss"),
        ("dpr", "memory"),
        ("embedding", "elasticsearch"),
        ("embedding", "faiss"),
        ("embedding", "memory"),
        ("bm25", "elasticsearch"),
        ("bm25", "memory"),
        ("bm25", "weaviate"),
        ("es_filter_only", "elasticsearch"),
        ("tfidf", "memory"),
    ],
    indirect=True,
)
def test_retrieval_without_filters(retriever_with_docs: BaseRetriever, document_store_with_docs: BaseDocumentStore):
    if not isinstance(retriever_with_docs, (BM25Retriever, TfidfRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    # NOTE: FilterRetriever simply returns all documents matching a filter,
    # so without filters applied it does nothing
    if not isinstance(retriever_with_docs, FilterRetriever):
        # the BM25 implementation in Weaviate would NOT pick up the expected records
        # because of the lack of stemming: "Who lives in berlin" returns only 1 record while
        # "Who live in berlin" returns all 5 records.
        # TODO - In Weaviate 1.19.0 there is a fix for the lack of stemming, which means that once 1.19.0 is released
        # this `if` can be removed, as the standard search query "Who lives in Berlin?" should work with Weaviate.
        # See https://github.com/weaviate/weaviate/issues/2439
        if isinstance(document_store_with_docs, WeaviateDocumentStore):
            res = retriever_with_docs.retrieve(query="Who live in berlin")
        else:
            res = retriever_with_docs.retrieve(query="Who lives in Berlin?")
        assert res[0].content == "My name is Carla and I live in Berlin"
        assert len(res) == 5
        assert res[0].meta["name"] == "filename1"


@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("mdr", "elasticsearch"),
        ("mdr", "memory"),
        ("dpr", "elasticsearch"),
        ("dpr", "memory"),
        ("embedding", "elasticsearch"),
        ("embedding", "memory"),
        ("bm25", "elasticsearch"),
        ("bm25", "weaviate"),
        ("es_filter_only", "elasticsearch"),
    ],
    indirect=True,
)
def test_retrieval_with_filters(retriever_with_docs: BaseRetriever, document_store_with_docs: BaseDocumentStore):
    if not isinstance(retriever_with_docs, (BM25Retriever, FilterRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

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


@pytest.mark.unit
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_tfidf_retriever_multiple_indexes(document_store: BaseDocumentStore):
    docs_index_0 = [Document(content="test_1"), Document(content="test_2"), Document(content="test_3")]
    docs_index_1 = [Document(content="test_4"), Document(content="test_5")]
    tfidf_retriever = TfidfRetriever(document_store=document_store)

    document_store.write_documents(docs_index_0, index="index_0")
    tfidf_retriever.fit(document_store, index="index_0")
    document_store.write_documents(docs_index_1, index="index_1")
    tfidf_retriever.fit(document_store, index="index_1")

    assert tfidf_retriever.document_counts["index_0"] == document_store.get_document_count(index="index_0")
    assert tfidf_retriever.document_counts["index_1"] == document_store.get_document_count(index="index_1")


@pytest.mark.unit
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_retrieval_empty_query(document_store: BaseDocumentStore):
    # test with empty query using the run() method
    mock_document = Document(id="0", content="test")
    retriever = MockBaseRetriever(document_store=document_store, mock_document=mock_document)
    result = retriever.run(root_node="Query", query="", filters={})
    assert result[0]["documents"][0] == mock_document

    result = retriever.run_batch(root_node="Query", queries=[""], filters={})
    assert result[0]["documents"][0][0] == mock_document


@pytest.mark.parametrize("retriever_with_docs", ["embedding", "dpr", "tfidf"], indirect=True)
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


@pytest.mark.parametrize("retriever_with_docs", ["embedding", "dpr", "tfidf"], indirect=True)
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


@pytest.mark.parametrize("retriever_with_docs", ["bm25"], indirect=True)
def test_batch_retrieval_multiple_queries_with_filters(retriever_with_docs, document_store_with_docs):
    if not isinstance(retriever_with_docs, (BM25Retriever, FilterRetriever)):
        document_store_with_docs.update_embeddings(retriever_with_docs)

    # Weaviate does not support BM25 with filters yet, only after Weaviate v1.18.0
    # TODO - remove this once Weaviate starts supporting BM25 WITH filters
    # You might also need to modify the first query, as Weaviate having problems with
    # retrieving the "My name is Carla and I live in Berlin" record just with the
    # "Who lives in Berlin?" BM25 query
    if isinstance(document_store_with_docs, WeaviateDocumentStore):
        return

    res = retriever_with_docs.retrieve_batch(
        queries=["Who lives in Berlin?", "Who lives in New York?"], filters=[{"name": "filename1"}, None]
    )

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


@pytest.mark.unit
def test_embed_meta_fields(docs_with_ids):
    with patch(
        "haystack.nodes.retriever._embedding_encoder._SentenceTransformersEmbeddingEncoder.__init__"
    ) as mock_init:
        mock_init.return_value = None
        retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers",
            embed_meta_fields=["date_field", "numeric_field", "list_field"],
        )
    docs_with_embedded_meta = retriever._preprocess_documents(docs=docs_with_ids[:2])
    assert docs_with_embedded_meta[0].content.startswith("2019-10-01\n5.0\nitem0.1\nitem0.2")
    assert docs_with_embedded_meta[1].content.startswith("2020-03-01\n5.5\nitem1.1\nitem1.2")


@pytest.mark.unit
def test_embed_meta_fields_empty():
    doc = Document(content="My name is Matteo and I live in Rome", meta={"meta_field": "", "list_field": []})
    with patch(
        "haystack.nodes.retriever._embedding_encoder._SentenceTransformersEmbeddingEncoder.__init__"
    ) as mock_init:
        mock_init.return_value = None
        retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers",
            embed_meta_fields=["meta_field", "list_field"],
        )
    docs_with_embedded_meta = retriever._preprocess_documents(docs=[doc])
    assert docs_with_embedded_meta[0].content == "My name is Matteo and I live in Rome"


@pytest.mark.unit
def test_embed_meta_fields_list_with_one_item():
    doc = Document(content="My name is Matteo and I live in Rome", meta={"list_field": ["one_item"]})
    with patch(
        "haystack.nodes.retriever._embedding_encoder._SentenceTransformersEmbeddingEncoder.__init__"
    ) as mock_init:
        mock_init.return_value = None
        retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            model_format="sentence_transformers",
            embed_meta_fields=["list_field"],
        )
    docs_with_embedded_meta = retriever._preprocess_documents(docs=[doc])
    assert docs_with_embedded_meta[0].content == "one_item\nMy name is Matteo and I live in Rome"


@pytest.mark.unit
def test_custom_query():
    mock_document_store = Mock(spec=KeywordDocumentStore)
    mock_document_store.index = "test"

    custom_query = """
            {
                "size": 10,
                "query": {
                    "bool": {
                        "should": [{
                            "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["custom_text_field"]}}],
                            "filter": ${filters}}}}"""

    retriever = BM25Retriever(document_store=mock_document_store, custom_query=custom_query)
    retriever.retrieve(query="test", filters={"year": ["2020", "2021"]})
    assert mock_document_store.query.call_args.kwargs["custom_query"] == custom_query
    assert mock_document_store.query.call_args.kwargs["filters"] == {"year": ["2020", "2021"]}
    assert mock_document_store.query.call_args.kwargs["query"] == "test"


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "weaviate", "pinecone"], indirect=True)
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
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "weaviate", "pinecone"], indirect=True)
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


@pytest.mark.unit
def test_openai_embedding_retriever_model_format():
    # support text-embedding-ada-002
    assert (
        EmbeddingRetriever._infer_model_format(model_name_or_path="text-embedding-ada-002", use_auth_token=None)
        == "openai"
    )

    # support old ada and other text-search-<modelname>-*-001 models
    assert EmbeddingRetriever._infer_model_format(model_name_or_path="ada", use_auth_token=None) == "openai"

    # support old babbage and other text-search-<modelname>-*-001 models
    assert EmbeddingRetriever._infer_model_format(model_name_or_path="babbage", use_auth_token=None) == "openai"

    # make sure that we can handle potential unreleased models
    assert (
        EmbeddingRetriever._infer_model_format(model_name_or_path="text-embedding-babbage-002", use_auth_token=None)
        == "openai"
    )


@pytest.mark.unit
def test_openai_encoder_setup_encoding_models():
    with patch("haystack.nodes.retriever._openai_encoder._OpenAIEmbeddingEncoder.__init__") as mock_encoder_init:
        mock_encoder_init.return_value = None
        encoder = _OpenAIEmbeddingEncoder(retriever=None)  # type: ignore

    encoder._setup_encoding_models(model_class="ada", model_name="text-embedding-ada-002", max_seq_len=512)
    assert encoder.query_encoder_model == "text-embedding-ada-002"
    assert encoder.doc_encoder_model == "text-embedding-ada-002"

    # support old ada and other text-search-<modelname>-*-001 models
    encoder._setup_encoding_models(model_class="ada", model_name="ada", max_seq_len=512)
    assert encoder.query_encoder_model == "text-search-ada-query-001"
    assert encoder.doc_encoder_model == "text-search-ada-doc-001"

    # support old babbage and other text-search-<modelname>-*-001 models
    encoder._setup_encoding_models(model_class="babbage", model_name="babbage", max_seq_len=512)
    assert encoder.query_encoder_model == "text-search-babbage-query-001"
    assert encoder.doc_encoder_model == "text-search-babbage-doc-001"

    # make sure that we can handle potential unreleased models
    encoder._setup_encoding_models(model_class="babbage", model_name="text-embedding-babbage-002", max_seq_len=512)
    assert encoder.query_encoder_model == "text-embedding-babbage-002"
    assert encoder.doc_encoder_model == "text-embedding-babbage-002"


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["cohere"], indirect=True)
@pytest.mark.embedding_dim(1024)
@pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY", None),
    reason="Please export an env var called COHERE_API_KEY containing " "the Cohere API key to run this test.",
)
def test_basic_cohere_embedding(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs = sorted(docs, key=lambda d: d.id)

    for doc in docs:
        assert len(doc.embedding) == 1024


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["openai"], indirect=True)
@pytest.mark.embedding_dim(1536)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason=("Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test."),
)
def test_basic_openai_embedding(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs = sorted(docs, key=lambda d: d.id)

    for doc in docs:
        assert len(doc.embedding) == 1536


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["azure"], indirect=True)
@pytest.mark.embedding_dim(1536)
@pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_API_KEY", None)
    and not os.environ.get("AZURE_OPENAI_BASE_URL", None)
    and not os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED", None),
    reason=(
        "Please export env variables called AZURE_OPENAI_API_KEY containing "
        "the Azure OpenAI key, AZURE_OPENAI_BASE_URL containing "
        "the Azure OpenAI base URL, and AZURE_OPENAI_DEPLOYMENT_NAME_EMBED containing "
        "the Azure OpenAI deployment name to run this test."
    ),
)
def test_basic_azure_embedding(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    docs = document_store.get_all_documents()
    docs = sorted(docs, key=lambda d: d.id)

    for doc in docs:
        assert len(doc.embedding) == 1536


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["cohere"], indirect=True)
@pytest.mark.embedding_dim(1024)
@pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY", None),
    reason="Please export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
)
def test_retriever_basic_cohere_search(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Madrid", params={"Retriever": {"top_k": 1}})
    assert len(res["documents"]) == 1
    assert "Madrid" in res["documents"][0].content


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["openai"], indirect=True)
@pytest.mark.embedding_dim(1536)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export env called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_retriever_basic_openai_search(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Madrid", params={"Retriever": {"top_k": 1}})
    assert len(res["documents"]) == 1
    assert "Madrid" in res["documents"][0].content


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["azure"], indirect=True)
@pytest.mark.embedding_dim(1536)
@pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_API_KEY", None)
    and not os.environ.get("AZURE_OPENAI_BASE_URL", None)
    and not os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED", None),
    reason=(
        "Please export env variables called AZURE_OPENAI_API_KEY containing "
        "the Azure OpenAI key, AZURE_OPENAI_BASE_URL containing "
        "the Azure OpenAI base URL, and AZURE_OPENAI_DEPLOYMENT_NAME_EMBED containing "
        "the Azure OpenAI deployment name to run this test."
    ),
)
def test_retriever_basic_azure_search(document_store, retriever, docs_with_ids):
    document_store.return_embedding = True
    document_store.write_documents(docs_with_ids)
    document_store.update_embeddings(retriever=retriever)

    p_retrieval = DocumentSearchPipeline(retriever)
    res = p_retrieval.run(query="Madrid", params={"Retriever": {"top_k": 1}})
    assert len(res["documents"]) == 1
    assert "Madrid" in res["documents"][0].content


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.parametrize("document_store", ["elasticsearch", "memory"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_text_retriever_embedding(document_store, retriever, docs):
    # BM25 representation is incompatible with table retriever
    if isinstance(document_store, InMemoryDocumentStore):
        document_store.use_bm25 = False

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


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_text_retriever_embedding_only_text(document_store, retriever):
    docs = [
        Document(content="This is a test", content_type="text"),
        Document(content="This is another test", content_type="text"),
    ]
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)


@pytest.mark.integration
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_table_text_retriever_embedding_only_table(document_store, retriever):
    doc = Document(
        content=pd.DataFrame(columns=["id", "text"], data=[["1", "This is a test"], ["2", "This is another test"]]),
        content_type="table",
    )
    document_store.write_documents([doc])
    document_store.update_embeddings(retriever)


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
    assert isinstance(loaded_retriever.passage_tokenizer, PreTrainedTokenizerFast)
    assert isinstance(loaded_retriever.query_tokenizer, PreTrainedTokenizerFast)
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
    assert isinstance(loaded_retriever.passage_tokenizer, PreTrainedTokenizerFast)
    assert isinstance(loaded_retriever.table_tokenizer, PreTrainedTokenizerFast)
    assert isinstance(loaded_retriever.query_tokenizer, PreTrainedTokenizerFast)
    assert loaded_retriever.passage_tokenizer.do_lower_case == True
    assert loaded_retriever.table_tokenizer.do_lower_case == True
    assert loaded_retriever.query_tokenizer.do_lower_case == True
    assert loaded_retriever.passage_tokenizer.vocab_size == 30522
    assert loaded_retriever.table_tokenizer.vocab_size == 30522
    assert loaded_retriever.query_tokenizer.vocab_size == 30522


@pytest.mark.embedding_dim(128)
def test_table_text_retriever_training(tmp_path, document_store, samples_path):
    retriever = TableTextRetriever(
        document_store=document_store,
        query_embedding_model="deepset/bert-small-mm_retrieval-question_encoder",
        passage_embedding_model="deepset/bert-small-mm_retrieval-passage_encoder",
        table_embedding_model="deepset/bert-small-mm_retrieval-table_encoder",
        use_gpu=False,
    )

    retriever.train(
        data_dir=samples_path / "mmr",
        train_filename="sample.json",
        n_epochs=1,
        n_gpu=0,
        save_dir=f"{tmp_path}/test_table_text_retriever_train",
    )

    # Load trained model
    retriever = TableTextRetriever.load(
        load_dir=f"{tmp_path}/test_table_text_retriever_train", document_store=document_store
    )


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


@pytest.mark.elasticsearch
def test_bm25retriever_all_terms_must_match():
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
    retriever = BM25Retriever(document_store=doc_store)
    results_wo_all_terms_must_match = retriever.retrieve(query="drink green tea")
    assert len(results_wo_all_terms_must_match) == 4
    retriever = BM25Retriever(document_store=doc_store, all_terms_must_match=True)
    results_w_all_terms_must_match = retriever.retrieve(query="drink green tea")
    assert len(results_w_all_terms_must_match) == 1
    retriever = BM25Retriever(document_store=doc_store)
    results_w_all_terms_must_match = retriever.retrieve(query="drink green tea", all_terms_must_match=True)
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


#
# MultiModal
#


@pytest.fixture
def text_docs() -> List[Document]:
    return [
        Document(
            content="My name is Paul and I live in New York",
            meta={
                "meta_field": "test2",
                "name": "filename2",
                "date_field": "2019-10-01",
                "numeric_field": 5.0,
                "odd_field": 0,
            },
        ),
        Document(
            content="My name is Carla and I live in Berlin",
            meta={
                "meta_field": "test1",
                "name": "filename1",
                "date_field": "2020-03-01",
                "numeric_field": 5.5,
                "odd_field": 1,
            },
        ),
        Document(
            content="My name is Christelle and I live in Paris",
            meta={
                "meta_field": "test3",
                "name": "filename3",
                "date_field": "2018-10-01",
                "numeric_field": 4.5,
                "odd_field": 1,
            },
        ),
        Document(
            content="My name is Camila and I live in Madrid",
            meta={
                "meta_field": "test4",
                "name": "filename4",
                "date_field": "2021-02-01",
                "numeric_field": 3.0,
                "odd_field": 0,
            },
        ),
        Document(
            content="My name is Matteo and I live in Rome",
            meta={
                "meta_field": "test5",
                "name": "filename5",
                "date_field": "2019-01-01",
                "numeric_field": 0.0,
                "odd_field": 1,
            },
        ),
    ]


@pytest.fixture
def table_docs() -> List[Document]:
    return [
        Document(
            content=pd.DataFrame(
                {
                    "Mountain": ["Mount Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"],
                    "Height": ["8848m", "8,611 m", "8 586m", "8 516 m", "8,485m"],
                }
            ),
            content_type="table",
        ),
        Document(
            content=pd.DataFrame(
                {
                    "City": ["Paris", "Lyon", "Marseille", "Lille", "Toulouse", "Bordeaux"],
                    "Population": ["13,114,718", "2,280,845", "1,873,270 ", "1,510,079", "1,454,158", "1,363,711"],
                }
            ),
            content_type="table",
        ),
        Document(
            content=pd.DataFrame(
                {
                    "City": ["Berlin", "Hamburg", "Munich", "Cologne"],
                    "Population": ["3,644,826", "1,841,179", "1,471,508", "1,085,664"],
                }
            ),
            content_type="table",
        ),
    ]


@pytest.fixture
def image_docs(samples_path) -> List[Document]:
    return [
        Document(content=str(samples_path / "images" / imagefile), content_type="image")
        for imagefile in os.listdir(samples_path / "images")
    ]


@pytest.mark.integration
def test_multimodal_text_retrieval(text_docs: List[Document]):
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True),
        query_embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        document_embedding_models={"text": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
    )
    retriever.document_store.write_documents(text_docs)
    retriever.document_store.update_embeddings(retriever=retriever)

    results = retriever.retrieve(query="Who lives in Paris?")
    assert results[0].content == "My name is Christelle and I live in Paris"


@pytest.mark.integration
def test_multimodal_text_retrieval_batch(text_docs: List[Document]):
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True),
        query_embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        document_embedding_models={"text": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
    )
    retriever.document_store.write_documents(text_docs)
    retriever.document_store.update_embeddings(retriever=retriever)

    results = retriever.retrieve_batch(queries=["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Madrid?"])
    assert results[0][0].content == "My name is Christelle and I live in Paris"
    assert results[1][0].content == "My name is Carla and I live in Berlin"
    assert results[2][0].content == "My name is Camila and I live in Madrid"


@pytest.mark.integration
def test_multimodal_table_retrieval(table_docs: List[Document]):
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True),
        query_embedding_model="deepset/all-mpnet-base-v2-table",
        document_embedding_models={"table": "deepset/all-mpnet-base-v2-table"},
    )
    retriever.document_store.write_documents(table_docs)
    retriever.document_store.update_embeddings(retriever=retriever)

    results = retriever.retrieve(query="How many people live in Hamburg?")
    assert_frame_equal(
        results[0].content,
        pd.DataFrame(
            {
                "City": ["Berlin", "Hamburg", "Munich", "Cologne"],
                "Population": ["3,644,826", "1,841,179", "1,471,508", "1,085,664"],
            }
        ),
    )


@pytest.mark.skip("Must be reworked as it fails randomly")
@pytest.mark.integration
def test_multimodal_retriever_query():
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True, embedding_dim=512),
        query_embedding_model="sentence-transformers/clip-ViT-B-32",
        document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
    )

    res_emb = retriever.embed_queries(["dummy query 1", "dummy query 1"])
    assert np.array_equal(res_emb[0], res_emb[1])


@pytest.mark.integration
def test_multimodal_image_retrieval(image_docs: List[Document], samples_path):
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True, embedding_dim=512),
        query_embedding_model="sentence-transformers/clip-ViT-B-32",
        document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
    )
    retriever.document_store.write_documents(image_docs)
    retriever.document_store.update_embeddings(retriever=retriever)

    results = retriever.retrieve(query="What's a cat?")
    assert str(results[0].content) == str(samples_path / "images" / "cat.jpg")


@pytest.mark.skip("Not working yet as intended")
@pytest.mark.integration
def test_multimodal_text_image_retrieval(text_docs: List[Document], image_docs: List[Document], samples_path):
    retriever = MultiModalRetriever(
        document_store=InMemoryDocumentStore(return_embedding=True, embedding_dim=512),
        query_embedding_model="sentence-transformers/clip-ViT-B-32",
        document_embedding_models={
            "text": "sentence-transformers/clip-ViT-B-32",
            "image": "sentence-transformers/clip-ViT-B-32",
        },
    )
    retriever.document_store.write_documents(image_docs)
    retriever.document_store.write_documents(text_docs)
    retriever.document_store.update_embeddings(retriever=retriever)

    results = retriever.retrieve(query="What's Paris?")

    text_results = [result for result in results if result.content_type == "text"]
    image_results = [result for result in results if result.content_type == "image"]

    assert str(image_results[0].content) == str(samples_path / "images" / "paris.jpg")
    assert text_results[0].content == "My name is Christelle and I live in Paris"


@pytest.mark.unit
@patch("haystack.nodes.retriever._openai_encoder.openai_request")
def test_openai_default_api_base(mock_request):
    with patch("haystack.nodes.retriever._openai_encoder.load_openai_tokenizer"):
        retriever = EmbeddingRetriever(embedding_model="text-embedding-ada-002", api_key="fake_api_key")
    assert retriever.api_base == "https://api.openai.com/v1"

    retriever.embed_queries(queries=["test query"])
    assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/embeddings"
    mock_request.reset_mock()

    retriever.embed_documents(documents=[Document(content="test document")])
    assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/embeddings"


@pytest.mark.unit
@patch("haystack.nodes.retriever._openai_encoder.openai_request")
def test_openai_custom_api_base(mock_request):
    with patch("haystack.nodes.retriever._openai_encoder.load_openai_tokenizer"):
        retriever = EmbeddingRetriever(
            embedding_model="text-embedding-ada-002", api_key="fake_api_key", api_base="https://fake_api_base.com"
        )
    assert retriever.api_base == "https://fake_api_base.com"

    retriever.embed_queries(queries=["test query"])
    assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/embeddings"
    mock_request.reset_mock()

    retriever.embed_documents(documents=[Document(content="test document")])
    assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/embeddings"


@pytest.mark.unit
@patch("haystack.nodes.retriever._openai_encoder.openai_request")
def test_openai_no_openai_organization(mock_request):
    with patch("haystack.nodes.retriever._openai_encoder.load_openai_tokenizer"):
        retriever = EmbeddingRetriever(embedding_model="text-embedding-ada-002", api_key="fake_api_key")
    assert retriever.openai_organization is None

    retriever.embed_queries(queries=["test query"])
    assert "OpenAI-Organization" not in mock_request.call_args.kwargs["headers"]


@pytest.mark.unit
@patch("haystack.nodes.retriever._openai_encoder.openai_request")
def test_openai_openai_organization(mock_request):
    with patch("haystack.nodes.retriever._openai_encoder.load_openai_tokenizer"):
        retriever = EmbeddingRetriever(
            embedding_model="text-embedding-ada-002", api_key="fake_api_key", openai_organization="fake_organization"
        )
    assert retriever.openai_organization == "fake_organization"

    retriever.embed_queries(queries=["test query"])
    assert mock_request.call_args.kwargs["headers"]["OpenAI-Organization"] == "fake_organization"
