import os
from abc import abstractmethod

import pytest

from haystack import Document
from haystack.nodes.retriever import MultiModalRetriever
from haystack.nodes.retriever.base import BaseRetriever
from haystack.document_stores import InMemoryDocumentStore

from test.nodes.retrievers.base import ABC_TestRetriever, ABC_TestTextRetriever
from test.nodes.retrievers.test_tabletext import ABC_TestTableRetriever
from test.conftest import SAMPLES_PATH


class ABC_TestImageRetriever(ABC_TestRetriever):
    """
    Base class for the suites of all Retrievers that can handle images.
    """

    @pytest.fixture
    def image_docs(self):
        return [
            Document(content=SAMPLES_PATH / "images" / imagefile, content_type="image")
            for imagefile in os.listdir(SAMPLES_PATH / "images")
        ]

    @abstractmethod
    @pytest.fixture()
    def image_retriever(self):
        raise NotImplementedError("Abstract method, use a subclass")

    @pytest.mark.integration
    def test_image_retrieval(self, image_retriever: BaseRetriever):
        results = image_retriever.retrieve(query="cat")
        assert results[0].content == SAMPLES_PATH / "images" / "cat.jpg"


class TestMultiModalRetriever(ABC_TestTextRetriever, ABC_TestTableRetriever, ABC_TestImageRetriever):
    @pytest.fixture()
    def retriever(self, docs):
        retriever = MultiModalRetriever(
            document_store=InMemoryDocumentStore(return_embedding=True),
            query_embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            document_embedding_models={"text": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
        )
        retriever.document_store.write_documents(docs)
        retriever.document_store.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def image_retriever(self, image_docs):
        retriever = MultiModalRetriever(
            document_store=InMemoryDocumentStore(return_embedding=True, embedding_dim=512),
            query_embedding_model="sentence-transformers/clip-ViT-B-32",
            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        )
        retriever.document_store.write_documents(image_docs)
        retriever.document_store.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def table_retriever(self, table_docs):
        retriever = MultiModalRetriever(
            document_store=InMemoryDocumentStore(return_embedding=True),
            query_embedding_model="deepset/all-mpnet-base-v2-table",
            document_embedding_models={"table": "deepset/all-mpnet-base-v2-table"},
        )
        retriever.document_store.write_documents(table_docs)
        retriever.document_store.update_embeddings(retriever=retriever)
        return retriever
