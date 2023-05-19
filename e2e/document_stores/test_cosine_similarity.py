from copy import deepcopy
import math

import pytest
import numpy as np

from haystack.schema import Document

from ..conftest import document_store


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


@pytest.mark.parametrize("name", ["faiss", "weaviate", "opensearch_faiss", "elasticsearch", "memory"])
def test_cosine_similarity(name, tmp_path):
    documents = [Document.from_dict(d) for d in DOCUMENTS]
    with document_store(name, documents, tmp_path) as ds:
        # below we will write documents to the store and then query it to see if vectors were normalized or not
        query = np.random.rand(768).astype(np.float32)
        query_results = ds.query_by_embedding(
            query_emb=query, top_k=len(documents), return_embedding=True, scale_score=False
        )

        # check if search with cosine similarity returns the correct number of results
        assert len(query_results) == len(documents)

        original_embeddings = {doc["content"]: doc["embedding"] for doc in DOCUMENTS}

        for doc in query_results:
            result_emb = doc.embedding
            original_emb = original_embeddings[doc.content]

            expected_emb = original_emb
            # embeddings of document stores which only support dot product out of the box must be normalized
            if name in ["faiss", "weaviate", "opensearch_faiss"]:
                expected_emb = original_emb / np.linalg.norm(original_emb)

            # check if the stored embedding was normalized or not
            np.testing.assert_allclose(
                expected_emb, result_emb, rtol=0.2, atol=5e-07
            )  # high tolerance was necessary for Milvus 2

            # check if the score is plausible for cosine similarity
            cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
            assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize("name", ["faiss", "weaviate", "opensearch_faiss", "elasticsearch", "memory"])
def test_update_embeddings_cosine_similarity(name, tmp_path):
    # clear embeddings and convert to Document
    documents = deepcopy(DOCUMENTS)
    for doc in documents:
        doc.pop("embedding")
    documents = [Document.from_dict(d) for d in documents]

    with document_store(name, documents, tmp_path) as ds:
        # we wrote documents to the store and then query it to see if vectors were normalized
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
        ds.update_embeddings(retriever=retriever)

        query = np.random.rand(768).astype(np.float32)
        query_results = ds.query_by_embedding(
            query_emb=query, top_k=len(DOCUMENTS), return_embedding=True, scale_score=False
        )

        # check if search with cosine similarity returns the correct number of results
        assert len(query_results) == len(DOCUMENTS)

        for doc in query_results:
            result_emb = doc.embedding
            original_emb = original_embeddings[doc.content]

            expected_emb = original_emb
            # embeddings of document stores which only support dot product out of the box must be normalized
            if name in ["faiss", "weaviate", "opensearch_faiss"]:
                expected_emb = original_emb / np.linalg.norm(original_emb)

            # check if the stored embedding was normalized or not
            np.testing.assert_allclose(
                expected_emb, result_emb, rtol=0.2, atol=5e-07
            )  # high tolerance was necessary for Milvus 2

            # check if the score is plausible for cosine similarity
            cosine_score = np.dot(result_emb, query) / (np.linalg.norm(result_emb) * np.linalg.norm(query))
            assert cosine_score == pytest.approx(doc.score, 0.01)


@pytest.mark.parametrize("name", ["faiss", "weaviate", "memory", "elasticsearch", "opensearch_faiss"])
def test_cosine_sanity_check(name, tmp_path):
    VEC_1 = np.array([0.1, 0.2, 0.3], dtype="float32")
    VEC_2 = np.array([0.4, 0.5, 0.6], dtype="float32")

    # This is the cosine similarity of VEC_1 and VEC_2 calculated using sklearn.metrics.pairwise.cosine_similarity
    # The score is normalized to yield a value between 0 and 1.
    KNOWN_COSINE = 0.9746317
    KNOWN_SCALED_COSINE = (KNOWN_COSINE + 1) / 2

    docs = [Document.from_dict({"name": "vec_1", "text": "vec_1", "content": "vec_1", "embedding": VEC_1})]
    with document_store(name, docs, tmp_path, embedding_dim=3) as ds:
        query_results = ds.query_by_embedding(query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=True)

        # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
        assert math.isclose(query_results[0].score, KNOWN_SCALED_COSINE, abs_tol=0.0002)

        query_results = ds.query_by_embedding(query_emb=VEC_2, top_k=1, return_embedding=True, scale_score=False)

        # check if faiss returns the same cosine similarity. Manual testing with faiss yielded 0.9746318
        assert math.isclose(query_results[0].score, KNOWN_COSINE, abs_tol=0.0002)
