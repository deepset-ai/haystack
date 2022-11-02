import pytest

from haystack.document_stores.sql import SQLDocumentStore
from haystack.schema import Document

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

    @pytest.mark.integration
    def test_sql_write_document_invalid_meta(self, ds):
        documents = [
            {
                "content": "dict_with_invalid_meta",
                "valid_meta_field": "test1",
                "invalid_meta_field": [1, 2, 3],
                "name": "filename1",
                "id": "1",
            },
            Document(
                content="document_object_with_invalid_meta",
                meta={"valid_meta_field": "test2", "invalid_meta_field": [1, 2, 3], "name": "filename2"},
                id="2",
            ),
        ]
        ds.write_documents(documents)
        documents_in_store = ds.get_all_documents()
        assert len(documents_in_store) == 2
        assert ds.get_document_by_id("1").meta == {"name": "filename1", "valid_meta_field": "test1"}
        assert ds.get_document_by_id("2").meta == {"name": "filename2", "valid_meta_field": "test2"}

    @pytest.mark.integration
    def test_sql_write_different_documents_same_vector_id(self, ds):
        doc1 = {"content": "content 1", "name": "doc1", "id": "1", "vector_id": "vector_id"}
        doc2 = {"content": "content 2", "name": "doc2", "id": "2", "vector_id": "vector_id"}

        ds.write_documents([doc1], index="index1")
        documents_in_index1 = ds.get_all_documents(index="index1")
        assert len(documents_in_index1) == 1
        ds.write_documents([doc2], index="index2")
        documents_in_index2 = ds.get_all_documents(index="index2")
        assert len(documents_in_index2) == 1

        ds.write_documents([doc1], index="index3")
        with pytest.raises(Exception, match=r"(?i)unique"):
            ds.write_documents([doc2], index="index3")

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
