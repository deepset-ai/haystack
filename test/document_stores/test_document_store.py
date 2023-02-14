from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock


from ..conftest import get_document_store, ensure_ids_are_correct_uuids
from haystack.document_stores import (
    InMemoryDocumentStore,
    WeaviateDocumentStore,
    MilvusDocumentStore,
    FAISSDocumentStore,
    ElasticsearchDocumentStore,
    OpenSearchDocumentStore,
)

from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.es_converter import elasticsearch_index_to_document_store
from haystack.schema import Document, Label, Answer, Span
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.pipelines import DocumentSearchPipeline


DOCUMENTS = [
    {
        "meta": {"name": "name_1", "year": "2020", "month": "01"},
        "content": "text_1",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_2", "year": "2020", "month": "02"},
        "content": "text_2",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_3", "year": "2020", "month": "03"},
        "content": "text_3",
        "embedding": np.random.rand(768).astype(np.float64),
    },
    {
        "meta": {"name": "name_4", "year": "2021", "month": "01"},
        "content": "text_4",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_5", "year": "2021", "month": "02"},
        "content": "text_5",
        "embedding": np.random.rand(768).astype(np.float32),
    },
    {
        "meta": {"name": "name_6", "year": "2021", "month": "03"},
        "content": "text_6",
        "embedding": np.random.rand(768).astype(np.float64),
    },
]


@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory", "milvus", "weaviate"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
def test_update_embeddings(document_store, retriever):
    documents = []
    for i in range(6):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    documents.append({"content": "text_0", "id": "6", "meta_field": "value_0"})

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever, batch_size=3)
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 7
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    documents = document_store.get_all_documents(filters={"meta_field": ["value_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    documents = document_store.get_all_documents(filters={"meta_field": ["value_0", "value_5"]}, return_embedding=True)
    documents_with_value_0 = [doc for doc in documents if doc.meta["meta_field"] == "value_0"]
    documents_with_value_5 = [doc for doc in documents if doc.meta["meta_field"] == "value_5"]
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        documents_with_value_0[0].embedding,
        documents_with_value_5[0].embedding,
    )

    doc = {
        "content": "text_7",
        "id": "7",
        "meta_field": "value_7",
        "embedding": retriever.embed_queries(queries=["a random string"])[0],
    }
    document_store.write_documents([doc])

    documents = []
    for i in range(8, 11):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    document_store.write_documents(documents)

    doc_before_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
    embedding_before_update = doc_before_update.embedding

    # test updating only documents without embeddings
    if not isinstance(document_store, WeaviateDocumentStore):
        # All the documents in Weaviate store have an embedding by default. "update_existing_embeddings=False" is not allowed
        document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

    # test updating with filters
    if isinstance(document_store, FAISSDocumentStore):
        with pytest.raises(Exception):
            document_store.update_embeddings(
                retriever, update_existing_embeddings=True, filters={"meta_field": ["value"]}
            )
    else:
        document_store.update_embeddings(retriever, batch_size=3, filters={"meta_field": ["value_0", "value_1"]})
        doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
        embedding_after_update = doc_after_update.embedding
        np.testing.assert_array_equal(embedding_before_update, embedding_after_update)

    # test update all embeddings
    document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=True)
    assert document_store.get_embedding_count() == 11
    doc_after_update = document_store.get_all_documents(filters={"meta_field": ["value_7"]})[0]
    embedding_after_update = doc_after_update.embedding
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, embedding_before_update, embedding_after_update
    )

    # test update embeddings for newly added docs
    documents = []
    for i in range(12, 15):
        documents.append({"content": f"text_{i}", "id": str(i), "meta_field": f"value_{i}"})
    document_store.write_documents(documents)

    if not isinstance(document_store, WeaviateDocumentStore):
        # All the documents in Weaviate store have an embedding by default. "update_existing_embeddings=False" is not allowed
        document_store.update_embeddings(retriever, batch_size=3, update_existing_embeddings=False)
        assert document_store.get_embedding_count() == 14


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("retriever", ["table_text_retriever"], indirect=True)
@pytest.mark.embedding_dim(512)
def test_update_embeddings_table_text_retriever(document_store, retriever):
    documents = []
    for i in range(3):
        documents.append(
            {"content": f"text_{i}", "id": f"pssg_{i}", "meta_field": f"value_text_{i}", "content_type": "text"}
        )
        documents.append(
            {
                "content": pd.DataFrame(columns=[f"col_{i}", f"col_{i+1}"], data=[[f"cell_{i}", f"cell_{i+1}"]]),
                "id": f"table_{i}",
                f"meta_field": f"value_table_{i}",
                "content_type": "table",
            }
        )
    documents.append({"content": "text_0", "id": "pssg_4", "meta_field": "value_text_0", "content_type": "text"})
    documents.append(
        {
            "content": pd.DataFrame(columns=["col_0", "col_1"], data=[["cell_0", "cell_1"]]),
            "id": "table_4",
            "meta_field": "value_table_0",
            "content_type": "table",
        }
    )

    document_store.write_documents(documents)
    document_store.update_embeddings(retriever, batch_size=3)
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 8
    for doc in documents:
        assert type(doc.embedding) is np.ndarray

    # Check if Documents with same content (text) get same embedding
    documents = document_store.get_all_documents(filters={"meta_field": ["value_text_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_text_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    # Check if Documents with same content (table) get same embedding
    documents = document_store.get_all_documents(filters={"meta_field": ["value_table_0"]}, return_embedding=True)
    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["meta_field"] == "value_table_0"
    np.testing.assert_array_almost_equal(documents[0].embedding, documents[1].embedding, decimal=4)

    # Check if Documents wih different content (text) get different embedding
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_text_1", "value_text_2"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )

    # Check if Documents with different content (table) get different embeddings
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_table_1", "value_table_2"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )

    # Check if Documents with different content (table + text) get different embeddings
    documents = document_store.get_all_documents(
        filters={"meta_field": ["value_text_1", "value_table_1"]}, return_embedding=True
    )
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, documents[0].embedding, documents[1].embedding
    )


@pytest.mark.parametrize("document_store_type", ["elasticsearch", "memory"])
def test_custom_embedding_field(document_store_type, tmp_path):
    document_store = get_document_store(
        document_store_type=document_store_type,
        tmp_path=tmp_path,
        embedding_field="custom_embedding_field",
        index="custom_embedding_field",
    )
    doc_to_write = {"content": "test", "custom_embedding_field": np.random.rand(768).astype(np.float32)}
    document_store.write_documents([doc_to_write])
    documents = document_store.get_all_documents(return_embedding=True)
    assert len(documents) == 1
    assert documents[0].content == "test"
    np.testing.assert_array_equal(doc_to_write["custom_embedding_field"], documents[0].embedding)


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_get_meta_values_by_key(document_store: BaseDocumentStore):
    documents = [Document(content=f"Doc{i}", meta={"meta_key_1": f"{i}", "meta_key_2": f"{i}{i}"}) for i in range(20)]
    document_store.write_documents(documents)

    # test without filters or query
    result = document_store.get_metadata_values_by_key(key="meta_key_1")
    possible_values = [f"{i}" for i in range(20)]
    assert len(result) == 20
    for bucket in result:
        assert bucket["value"] in possible_values
        assert bucket["count"] == 1

    # test with filters but no query
    result = document_store.get_metadata_values_by_key(key="meta_key_1", filters={"meta_key_2": ["11", "22"]})
    for bucket in result:
        assert bucket["value"] in ["1", "2"]
        assert bucket["count"] == 1

    # test with filters & query
    result = document_store.get_metadata_values_by_key(key="meta_key_1", query="Doc1")
    for bucket in result:
        assert bucket["value"] in ["1"]
        assert bucket["count"] == 1


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_sentence_transformers(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs, embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert [document.content for document in prediction["documents"]] == [
        "My name is Paul and I live in New York",
        "My name is Matteo and I live in Rome",
        "My name is Christelle and I live in Paris",
        "My name is Carla and I live in Berlin",
        "My name is Camila and I live in Madrid",
    ]
    assert scores == pytest.approx(
        [0.9149981737136841, 0.6895168423652649, 0.641706794500351, 0.6206043660640717, 0.5837393924593925], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_format="farm",
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.9102507941407827, 0.6937791467877008, 0.6491682889305038, 0.6321622491318529, 0.5909129441370939], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_with_docs", ["memory", "faiss", "milvus", "weaviate", "elasticsearch"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_without_scaling(document_store_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        scale_score=False,
        model_format="farm",
    )
    document_store_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.8205015882815654, 0.3875582935754016, 0.29833657786100765, 0.26432449826370585, 0.18182588827418789],
        abs=1e-3,
    )


@pytest.mark.parametrize(
    "document_store_dot_product_with_docs", ["memory", "faiss", "milvus", "elasticsearch", "weaviate"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_dot_product(document_store_dot_product_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_dot_product_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_format="farm",
    )
    document_store_dot_product_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [0.5526494403409358, 0.5247784342375555, 0.5189836829440964, 0.5179697273254912, 0.5112024928228626], abs=1e-3
    )


@pytest.mark.parametrize(
    "document_store_dot_product_with_docs", ["memory", "faiss", "milvus", "elasticsearch", "weaviate"], indirect=True
)
@pytest.mark.embedding_dim(384)
def test_similarity_score_dot_product_without_scaling(document_store_dot_product_with_docs):
    retriever = EmbeddingRetriever(
        document_store=document_store_dot_product_with_docs,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        scale_score=False,
        model_format="farm",
    )
    document_store_dot_product_with_docs.update_embeddings(retriever)
    pipeline = DocumentSearchPipeline(retriever)
    prediction = pipeline.run("Paul lives in New York")
    scores = [document.score for document in prediction["documents"]]
    assert scores == pytest.approx(
        [21.13810000000001, 9.919499999999971, 7.597099999999955, 7.191000000000031, 4.481750000000034], abs=1e-3
    )


def test_custom_headers(document_store_with_docs: BaseDocumentStore):
    mock_client = None
    if isinstance(document_store_with_docs, ElasticsearchDocumentStore):
        es_document_store: ElasticsearchDocumentStore = document_store_with_docs
        mock_client = Mock(wraps=es_document_store.client)
        es_document_store.client = mock_client
    custom_headers = {"X-My-Custom-Header": "header-value"}
    if not mock_client:
        with pytest.raises(NotImplementedError):
            documents = document_store_with_docs.get_all_documents(headers=custom_headers)
    else:
        documents = document_store_with_docs.get_all_documents(headers=custom_headers)
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        assert "headers" in kwargs
        assert kwargs["headers"] == custom_headers
        assert len(documents) > 0


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_elasticsearch_brownfield_support(document_store_with_docs):
    new_document_store = InMemoryDocumentStore()
    new_document_store = elasticsearch_index_to_document_store(
        document_store=new_document_store,
        original_index_name="haystack_test",
        original_content_field="content",
        original_name_field="name",
        included_metadata_fields=["date_field"],
        index="test_brownfield_support",
        id_hash_keys=["content", "meta"],
    )

    original_documents = document_store_with_docs.get_all_documents(index="haystack_test")
    transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support")
    assert len(original_documents) == len(transferred_documents)
    assert all("name" in doc.meta for doc in transferred_documents)
    assert all("date_field" in doc.meta for doc in transferred_documents)
    assert all("meta_field" not in doc.meta for doc in transferred_documents)
    assert all("numeric_field" not in doc.meta for doc in transferred_documents)
    assert all(doc.id == doc._get_id(["content", "meta"]) for doc in transferred_documents)

    original_content = set([doc.content for doc in original_documents])
    transferred_content = set([doc.content for doc in transferred_documents])
    assert original_content == transferred_content

    # Test transferring docs with PreProcessor
    new_document_store = elasticsearch_index_to_document_store(
        document_store=new_document_store,
        original_index_name="haystack_test",
        original_content_field="content",
        excluded_metadata_fields=["date_field"],
        index="test_brownfield_support_2",
        preprocessor=PreProcessor(split_length=1, split_respect_sentence_boundary=False),
    )
    transferred_documents = new_document_store.get_all_documents(index="test_brownfield_support_2")
    assert all("date_field" not in doc.meta for doc in transferred_documents)
    assert all("name" in doc.meta for doc in transferred_documents)
    assert all("meta_field" in doc.meta for doc in transferred_documents)
    assert all("numeric_field" in doc.meta for doc in transferred_documents)
    # Check if number of transferred_documents is equal to number of unique words.
    assert len(transferred_documents) == len(set(" ".join(original_content).split()))


@pytest.mark.parametrize(
    "document_store", ["faiss", "milvus", "weaviate", "opensearch", "elasticsearch", "memory"], indirect=True
)
def test_cosine_similarity(document_store: BaseDocumentStore):
    # below we will write documents to the store and then query it to see if vectors were normalized or not
    ensure_ids_are_correct_uuids(docs=DOCUMENTS, document_store=document_store)
    document_store.write_documents(documents=DOCUMENTS)

    query = np.random.rand(768).astype(np.float32)
    query_results = document_store.query_by_embedding(
        query_emb=query, top_k=len(DOCUMENTS), return_embedding=True, scale_score=False
    )

    # check if search with cosine similarity returns the correct number of results
    assert len(query_results) == len(DOCUMENTS)

    original_embeddings = {doc["content"]: doc["embedding"] for doc in DOCUMENTS}

    for doc in query_results:
        result_emb = doc.embedding
        original_emb = original_embeddings[doc.content]

        expected_emb = original_emb
        # embeddings of document stores which only support dot product out of the box must be normalized
        if (
            isinstance(document_store, (FAISSDocumentStore, MilvusDocumentStore, WeaviateDocumentStore))
            or isinstance(document_store, OpenSearchDocumentStore)
            and document_store.knn_engine == "faiss"
        ):
            expected_emb = original_emb / np.linalg.norm(original_emb)

        # check if the stored embedding was normalized or not
        np.testing.assert_allclose(
            expected_emb, result_emb, rtol=0.2, atol=5e-07
        )  # high tolerance necessary for Milvus 2

        # check if the score is plausible for cosine similarity
        cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
        assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize(
    "document_store", ["faiss", "milvus", "weaviate", "opensearch", "elasticsearch", "memory"], indirect=True
)
def test_update_embeddings_cosine_similarity(document_store: BaseDocumentStore):
    # below we will write documents to the store and then query it to see if vectors were normalized
    ensure_ids_are_correct_uuids(docs=DOCUMENTS, document_store=document_store)
    # clear embeddings
    docs = deepcopy(DOCUMENTS)
    for doc in docs:
        doc.pop("embedding")

    document_store.write_documents(documents=docs)
    original_embeddings = {}

    # now check if vectors are normalized when updating embeddings
    class MockRetriever:
        def embed_documents(self, docs):
            embeddings = []
            for doc in docs:
                embedding = np.random.rand(768).astype(np.float32)
                original_embeddings[doc.content] = embedding
                embeddings.append(embedding)
            return np.stack(embeddings)

    retriever = MockRetriever()
    document_store.update_embeddings(retriever=retriever)

    query = np.random.rand(768).astype(np.float32)
    query_results = document_store.query_by_embedding(
        query_emb=query, top_k=len(DOCUMENTS), return_embedding=True, scale_score=False
    )

    # check if search with cosine similarity returns the correct number of results
    assert len(query_results) == len(DOCUMENTS)

    for doc in query_results:
        result_emb = doc.embedding
        original_emb = original_embeddings[doc.content]

        expected_emb = original_emb
        # embeddings of document stores which only support dot product out of the box must be normalized
        if (
            isinstance(document_store, (FAISSDocumentStore, MilvusDocumentStore, WeaviateDocumentStore))
            or isinstance(document_store, OpenSearchDocumentStore)
            and document_store.knn_engine == "faiss"
        ):
            expected_emb = original_emb / np.linalg.norm(original_emb)

        # check if the stored embedding was normalized or not
        np.testing.assert_allclose(
            expected_emb, result_emb, rtol=0.2, atol=5e-07
        )  # high tolerance necessary for Milvus 2

        # check if the score is plausible for cosine similarity
        cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
        assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize(
    "document_store_small", ["faiss", "milvus", "weaviate", "memory", "elasticsearch", "opensearch"], indirect=True
)
def test_cosine_sanity_check(document_store_small):
    VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32")
    VEC_2 = np.array([0.4, 0.5, 0.6], dtype="float32")

    # This is the cosine similarity of VEC_1 and VEC_2 calculated using sklearn.metrics.pairwise.cosine_similarity
    # The score is normalized to yield a value between 0 and 1.
    KNOWN_COSINE = 0.9746317
    KNOWN_SCALED_COSINE = (KNOWN_COSINE + 1) / 2

    docs = [{"name": "vec_1", "text": "vec_1", "content": "vec_1", "embedding": VEC_1}]
    ensure_ids_are_correct_uuids(docs=docs, document_store=document_store_small)
    document_store_small.write_documents(documents=docs)

    query_results = document_store_small.query_by_embedding(
        query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=True
    )

    # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
    assert math.isclose(query_results[0].score, KNOWN_SCALED_COSINE, abs_tol=0.0002)

    query_results = document_store_small.query_by_embedding(
        query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=False
    )

    # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
    assert math.isclose(query_results[0].score, KNOWN_COSINE, abs_tol=0.0002)
