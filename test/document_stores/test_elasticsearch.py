import pytest
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from .test_base import DocumentStoreTest


class TestElasticsearchDocumentStore(DocumentStoreTest):
    # Constants

    index_name = __name__

    @pytest.fixture
    def ds(self):
        """
        This fixture provides a working document store and takes care of removing the indices when done
        """
        labels_index_name = f"{self.index_name}_labels"
        ds = ElasticsearchDocumentStore(index=self.index_name, label_index=labels_index_name, create_index=True)
        yield ds
        ds.delete_index(self.index_name)
        ds.delete_index(labels_index_name)
