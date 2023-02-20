import json
import logging

import pytest

from haystack.document_stores.sql import LabelORM, SQLDocumentStore
from haystack.schema import Document
from haystack.testing import DocumentStoreBaseTestAbstract


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

    @pytest.mark.integration
    def test_sql_get_documents_using_nested_filters_about_classification(self, ds):
        documents = [
            Document(
                content="That's good. I like it.",
                id="1",
                meta={
                    "classification": {
                        "label": "LABEL_1",
                        "score": 0.694,
                        "details": {"LABEL_1": 0.694, "LABEL_0": 0.306},
                    }
                },
            ),
            Document(
                content="That's bad. I don't like it.",
                id="2",
                meta={
                    "classification": {
                        "label": "LABEL_0",
                        "score": 0.898,
                        "details": {"LABEL_0": 0.898, "LABEL_1": 0.102},
                    }
                },
            ),
        ]
        ds.write_documents(documents)

        assert ds.get_document_count() == 2
        assert len(ds.get_all_documents(filters={"classification.score": {"$gt": 0.1}})) == 2
        assert len(ds.get_all_documents(filters={"classification.label": ["LABEL_1", "LABEL_0"]})) == 2
        assert len(ds.get_all_documents(filters={"classification.score": {"$gt": 0.8}})) == 1
        assert len(ds.get_all_documents(filters={"classification.label": ["LABEL_1"]})) == 1
        assert len(ds.get_all_documents(filters={"classification.score": {"$gt": 0.95}})) == 0
        assert len(ds.get_all_documents(filters={"classification.label": ["LABEL_100"]})) == 0

    # NOTE: the SQLDocumentStore marshals metadata values with JSON so querying
    # using filters doesn't always work. While this should be considered a bug,
    # the relative tests are either customized or skipped while we work on a fix.

    @pytest.mark.integration
    def test_ne_filters(self, ds, caplog):
        with caplog.at_level(logging.WARNING):
            ds.get_all_documents(filters={"year": {"$ne": "2020"}})
            assert "filters won't work on metadata fields" in caplog.text

    @pytest.mark.integration
    def test_get_all_labels_legacy_document_id(self, ds):
        ds.session.add(
            LabelORM(
                id="123",
                no_answer=False,
                document=json.dumps(
                    {
                        "content": "Some content",
                        "content_type": "text",
                        "score": None,
                        "id": "fc18c987a8312e72a47fb1524f230bb0",
                        "meta": {},
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ),
                origin="user-feedback",
                query="Who made the PDF specification?",
                is_correct_answer=True,
                is_correct_document=True,
                answer=json.dumps(
                    {
                        "answer": "Adobe Systems",
                        "type": "extractive",
                        "context": "Some content",
                        "offsets_in_context": [{"start": 60, "end": 73}],
                        "offsets_in_document": [{"start": 60, "end": 73}],
                        # legacy document_id answer
                        "document_id": "fc18c987a8312e72a47fb1524f230bb0",
                        "meta": {},
                        "score": None,
                    }
                ),
                pipeline_id="some-123",
                index=ds.label_index,
            )
        )
        labels = ds.get_all_labels()
        assert labels[0].answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]

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

    @pytest.mark.skip(reason="embeddings are not supported")
    @pytest.mark.integration
    def test_get_embedding_count(self):
        pass

    @pytest.mark.skip(reason="embeddings are not supported")
    @pytest.mark.integration
    def test_custom_embedding_field(self, ds):
        pass
