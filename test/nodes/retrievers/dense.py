from typing import List

import logging
from math import isclose
from time import sleep
import subprocess
from abc import ABC, abstractmethod

import numpy as np
import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore


class _TestBaseRetriever(ABC):

    QUESTION = "Who lives in Berlin?"
    ANSWER = "My name is Carla and I live in Berlin"

    @pytest.fixture(scope="session")
    def init_elasticsearch(self, boot_wait: int = 15):
        logging.warning("Setting up ElasticSearch. This will add about %s seconds to the first test.", boot_wait)
        subprocess.run(
            [
                "docker start haystack_tests_elasticsearch || "
                'docker run -d -p 9200:9200 -e "discovery.type=single-node" '
                "--name haystack_tests_elasticsearch elasticsearch:7.9.2"
            ],
            shell=True,
            check=True,
        )
        sleep(boot_wait)
        yield
        subprocess.run(["docker stop haystack_tests_elasticsearch"], shell=True, check=True)

    @pytest.fixture
    def docstore(self, docs: List[Document], embedding_dim: int = 768):
        docstore = InMemoryDocumentStore(embedding_dim=embedding_dim)
        docstore.write_documents(docs)
        return docstore

    @abstractmethod
    @pytest.fixture()
    def test_retriever(self, docstore):
        raise NotImplementedError("Abstract method, use a subclass")

    #
    # End2End tests
    #

    def test_retrieval(self, test_retriever: BaseRetriever):
        res = test_retriever.retrieve(query="Who lives in Berlin?")
        assert len(res) > 0
        assert res[0].content == "My name is Carla and I live in Berlin"

    def test_retrieval_with_single_filter(self, test_retriever: BaseRetriever):
        result = test_retriever.retrieve(query="Who lives in Berlin?", filters={"name": ["filename1"]}, top_k=5)
        assert len(result) == 1
        assert result[0].content == "My name is Carla and I live in Berlin"
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter(self, test_retriever: BaseRetriever):
        result = test_retriever.retrieve(
            query="Who lives in Berlin?",
            filters={"name": ["filename5", "filename1"], "meta_field": ["test1", "test2"]},
            top_k=5,
        )
        assert len(result) == 1
        assert result[0].meta["name"] == "filename1"

    def test_retrieval_with_many_filter_no_match(self, test_retriever: BaseRetriever):
        result = test_retriever.retrieve(
            query="Who lives in Berlin?", filters={"name": ["filename1"], "meta_field": ["test2", "test3"]}, top_k=5
        )
        assert len(result) == 0

    def test_batch_retrieval(self, test_retriever: BaseRetriever):
        res = test_retriever.retrieve_batch(queries=["Who lives in Berlin?", "Who lives in Madrid?"], top_k=5)

        assert len(res) == 2  # 2 queries
        assert len(res[0]) == 5  # top_k = 5
        assert res[0][0].content == "My name is Carla and I live in Berlin"
        assert res[1][0].content == "My name is Camila and I live in Madrid"


class _TestSparseRetrievers(_TestBaseRetriever):
    pass


class _TestDenseRetrievers(_TestBaseRetriever):
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "document_store",
        ["elasticsearch", "faiss", "memory", "milvus1", "milvus", "weaviate", "pinecone"],
        indirect=True,
    )
    @pytest.mark.parametrize("retriever", ["retribert"], indirect=True)
    @pytest.mark.embedding_dim(128)
    def test_retribert_embedding(self, document_store, retriever, docs_with_ids):
        # if isinstance(document_store, WeaviateDocumentStore):
        #     # Weaviate sets the embedding dimension to 768 as soon as it is initialized.
        #     # We need 128 here and therefore initialize a new WeaviateDocumentStore.
        #     document_store = WeaviateDocumentStore(index="haystack_test", embedding_dim=128, recreate_index=True)
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
