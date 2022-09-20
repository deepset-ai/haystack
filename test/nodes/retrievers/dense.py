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

from test.nodes.retrievers.base import ABC_TestBaseRetriever


class ABC_TestDenseRetrievers(ABC_TestBaseRetriever):
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
