from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import InMemoryDocumentStore, BaseDocumentStore
from haystack.nodes.retriever.sparse import TfidfRetriever

from test.nodes.retrievers.base import ABC_TestRetriever


# FIXME Cannot inherit from ABC_TestTextRetrievers until we implement filters.
class TestTfidfRetriever(ABC_TestRetriever):
    @pytest.fixture
    def docstore(self, docs: List[Document], embedding_dim: int = 768):
        docstore = InMemoryDocumentStore(embedding_dim=embedding_dim)
        docstore.write_documents(docs)
        return docstore

    @pytest.fixture
    def retriever(self, docstore: BaseDocumentStore):
        return TfidfRetriever(document_store=docstore)

    def test_retrieval(self, retriever: BaseRetriever):
        res = retriever.retrieve(query="Who lives in Berlin?")
        assert len(res) > 0
        assert res[0].content == "My name is Carla and I live in Berlin"
