import pytest

from haystack.document_stores.sql import SQLDocumentStore

from .test_base import DocumentStoreBaseTestAbstract


class TestSQLDocumentStore(DocumentStoreBaseTestAbstract):
    # Constants

    index_name = __name__

    @pytest.fixture
    def ds(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/haystack_test.db"
        return SQLDocumentStore(url=db_url, index=self.index_name, isolation_level="AUTOCOMMIT")

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, SQLDocumentStore doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    # NOTE: the SQLDocumentStore behaves differently to the others when filters are applied.
    # While this should be considered a bug, the relative tests are skipped in the meantime

    @pytest.mark.skip
    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
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

    # NOTE: labels metadata are not supported

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        pass
