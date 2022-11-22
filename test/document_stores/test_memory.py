import logging

import pytest

from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.schema import Document

from .test_base import DocumentStoreBaseTestAbstract


class TestInMemoryDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def ds(self):
        return InMemoryDocumentStore(return_embedding=True, use_bm25=True)

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, this doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    @pytest.mark.skip
    @pytest.mark.integration
    def test_ne_filters(self, ds, caplog):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        pass

    @pytest.mark.integration
    def test_get_documents_by_id(self, ds, documents):
        """
        The base test uses the batch_size param that's not supported
        here, so we override the test case
        """
        ds.write_documents(documents)
        ids = [doc.id for doc in documents]
        result = {doc.id for doc in ds.get_documents_by_id(ids)}
        assert set(ids) == result
