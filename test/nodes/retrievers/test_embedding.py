from typing import List

import pytest
import logging
from math import isclose

import numpy as np

from haystack import Document
from haystack.document_stores import BaseDocumentStore, InMemoryDocumentStore
from haystack.nodes.retriever import EmbeddingRetriever

from test.nodes.retrievers.base import ABC_TestTextRetriever


class TestEmbeddingRetriever(ABC_TestTextRetriever):
    @pytest.fixture
    def docstore(self, docs_with_ids: List[Document]):
        docstore = InMemoryDocumentStore(return_embedding=True, similarity="cosine")
        docstore.write_documents(docs_with_ids)
        return docstore

    @pytest.fixture()
    def retriever(self, docstore: BaseDocumentStore):
        retriever = EmbeddingRetriever(document_store=docstore, embedding_model="deepset/sentence_bert", use_gpu=False)
        docstore.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def retribert_docstore(self, docstore: BaseDocumentStore, docs_with_ids: List[Document]):
        docstore = InMemoryDocumentStore(return_embedding=True, embedding_dim=128, similarity="cosine")
        docstore.write_documents(docs_with_ids)
        return docstore

    @pytest.fixture()
    def retribert_retriever(self, retribert_docstore: BaseDocumentStore):
        retriever = EmbeddingRetriever(
            document_store=retribert_docstore, embedding_model="yjernite/retribert-base-uncased", use_gpu=False
        )
        retribert_docstore.update_embeddings(retriever=retriever)
        return retriever

    def test_embeddings_encoder_of_embedding_retriever_should_warn_about_model_format(
        self, caplog, docstore: BaseDocumentStore
    ):
        with caplog.at_level(logging.WARNING):
            EmbeddingRetriever(
                document_store=docstore,
                embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_format="farm",
            )
            assert (
                "You may need to set model_format='sentence_transformers' to ensure correct loading of model."
                in caplog.text
            )

    def test_retribert_embedding(self, retribert_retriever: EmbeddingRetriever):
        sorted_docs = sorted(retribert_retriever.document_store.get_all_documents(), key=lambda d: d.id)

        expected_values = [0.14017, 0.05975, 0.14267, 0.15099, 0.14383]
        for doc, expected_value in zip(sorted_docs, expected_values):
            embedding = doc.embedding
            assert len(embedding) == 128
            # always normalize vector as faiss returns normalized vectors and other document stores do not
            embedding /= np.linalg.norm(embedding)
            assert isclose(embedding[0], expected_value, rel_tol=0.001)
