from typing import List

import pytest

from haystack.schema import Document
from haystack.nodes.retriever import MultiModalRetriever

from test.nodes.retrievers.base import ABC_TestTextRetrievers


class TestMultiModalRetriever(ABC_TestTextRetrievers):
    @pytest.fixture()
    def retriever(self, docstore):
        retriever = MultiModalRetriever(
            document_store=docstore,
            query_embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            document_embedding_models={"text": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
        )
        docstore.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def image_retriever(self, docstore):
        retriever = MultiModalRetriever(
            document_store=docstore,
            query_embedding_model="sentence-transformers/clip-ViT-B-32",
            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        )
        docstore.update_embeddings(retriever=retriever)
        return retriever

    @pytest.fixture()
    def table_retriever(self, docstore):
        retriever = MultiModalRetriever(
            document_store=docstore,
            query_embedding_model="deepset/all-mpnet-base-v2-table",
            document_embedding_models={"table": "deepset/all-mpnet-base-v2-table"},
        )
        docstore.update_embeddings(retriever=retriever)
        return retriever
