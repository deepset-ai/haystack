from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever.sparse import FilterRetriever

from test.nodes.retrievers.sparse import ABC_TestSparseRetrievers


@pytest.mark.elasticsearch
class TestFilterRetriever(ABC_TestSparseRetrievers):
    @pytest.fixture(autouse=True, scope="session")
    def init_docstore(self, init_elasticsearch):
        pass

    ## FIXME Why FilterRetriever inherits from BM25?
    ## FIXME This retriever should be able to work with all docstores, not only ES!
    ## FIXME Make FilterRetriever work at least on InMemoryDocumentStore
    @pytest.fixture
    def docstore(self, docs: List[Document]):
        docstore = ElasticsearchDocumentStore(
            index="haystack_test",
            return_embedding=True,
            embedding_dim=768,
            embedding_field="embedding",
            similarity="cosine",
            recreate_index=True,
        )
        docstore.write_documents(docs)
        yield docstore
        docstore.delete_documents()

    @pytest.fixture()
    def retriever(self, docstore):
        return FilterRetriever(document_store=docstore)

    def test_retrieval(self, retriever: BaseRetriever, docs: List[Document]):
        """
        Note: This test overrides the one in TestBaseRetriever.

        FilterRetriever is no-op when no filters are applied.
        It will simply return all documents (or as many as top_k allows) in the
        same order they arrive from the docstore.
        """
        res = retriever.retrieve(query="", top_k=len(docs) + 5)
        assert len(res) == len(docs)
        assert res == docs
