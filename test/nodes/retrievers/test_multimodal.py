from typing import List

import pytest

from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes.retriever import DensePassageRetriever

from test.nodes.retrievers.dense import TestDenseRetrievers


class TestMultiModalRetriever(TestDenseRetrievers):
    @pytest.fixture()
    def test_retriever(self, docstore):
        pass
