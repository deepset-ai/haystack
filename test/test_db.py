import pytest
import time

from haystack.database.base import Document


def test_get_all_documents(document_store_with_docs):
    documents = document_store_with_docs.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == 3
    assert {d.meta["name"] for d in documents} == {"filename1", "filename2", "filename3"}
    assert {d.meta["meta_field"] for d in documents} == {"test1", "test2", "test3"}
    doc = document_store_with_docs.get_document_by_id(documents[0].id)
    assert doc.id == documents[0].id
    assert doc.text == documents[0].text


@pytest.mark.parametrize("document_store_with_docs", [("elasticsearch")], indirect=True)
def test_elasticsearch_update_meta(document_store_with_docs):
    document = document_store_with_docs.query(query=None, filters={"name": ["filename1"]})[0]
    document_store_with_docs.update_document_meta(document.id, meta={"meta_field": "updated_meta"})
    updated_document = document_store_with_docs.query(query=None, filters={"name": ["filename1"]})[0]
    assert updated_document.meta["meta_field"] == "updated_meta"
