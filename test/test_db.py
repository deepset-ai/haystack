from haystack.database.sql import SQLDocumentStore
from haystack.indexing.io import write_documents_to_db


def test_db_write_read():
    sql_document_store = SQLDocumentStore()
    write_documents_to_db(document_store=sql_document_store, document_dir="samples/docs")
    documents = sql_document_store.get_all_documents()
    assert len(documents) == 2
    doc = sql_document_store.get_document_by_id("1")
    assert doc.keys() == {"id", "name", "text", "tags"}
