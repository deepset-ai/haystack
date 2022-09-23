import re
from typing import List
from urllib import request

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever.sparse import FilterRetriever

from test.nodes.retrievers.base import ABC_TestTextRetriever


class TestFilterRetriever(ABC_TestTextRetriever):
    """
    Supports:
     - N InMemory
     - Y Elasticsearch
     - Y OpenSearch
     - Y OpenDistroElasticsearch
     - Y SQL
     - Y FAISS
     - Y Milvus
     - Y Weaviate
     - Y Pinecone
    """

    ## FIXME Why FilterRetriever inherits from BM25?
    ## FIXME This retriever should be able to work with all docstores, not only ES!
    ## FIXME Make FilterRetriever work at least on InMemoryDocumentStore
    @pytest.fixture()
    def unsupported_docstores(self):
        return [
            "memory",  # Misses `query_batch`
            "faiss",  # Misses `query_batch`
            "milvus",  # Misses `query_batch`
            "weaviate",  # Misses `query_batch`
            "pinecone",  # Misses `query_batch`
        ]

    @pytest.fixture()
    def docs(self, docs_with_ids):
        return docs_with_ids

    @pytest.fixture()
    def retriever(self, docstore):
        return FilterRetriever(document_store=docstore)

    #
    # Tests
    #
    def test_retrieval(self, retriever: BaseRetriever, docs: List[Document]):
        """
        Note: This test overrides the one in TestBaseRetriever.

        FilterRetriever is no-op when no filters are applied.
        It will simply return all documents (or as many as top_k allows) in the
        same order they arrive from the docstore.
        """
        res = retriever.retrieve(query="", top_k=len(docs) + 5)
        assert sorted(res, key=lambda doc: doc.id) == sorted(docs, key=lambda doc: doc.id)
