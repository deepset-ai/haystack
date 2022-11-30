import pytest
import numpy as np

from haystack.document_stores.milvus import MilvusDocumentStore
from haystack.schema import Document

from .test_base import DocumentStoreBaseTestAbstract


class TestMilvusDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def ds(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/haystack_test_milvus.db"
        return MilvusDocumentStore(sql_url=db_url, return_embedding=True)

    @pytest.fixture
    def documents(self):
        """
        write_documents will raise an exception if receives a document without
        embeddings, so we customize the documents fixture and always provide
        embeddings
        """
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01", "numbers": [2, 4]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02", "numbers": [-2, -4]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i}",
                    meta={"name": f"name_{i}", "month": "03"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, MilvusDocumentStore doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    # NOTE: MilvusDocumentStore derives from the SQL one and behaves differently to the others when filters are applied.
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

    # NOTE: again inherithed from the SQLDocumentStore, labels metadata are not supported

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_filter_aggregations(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_meta_aggregations(self):
        pass
