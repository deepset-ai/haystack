from haystack.database.sql import SQLDocumentStore
from haystack.indexing.io import write_documents_to_db


def test_db_write_read():
    sql_datastore = SQLDocumentStore()
    write_documents_to_db(datastore=sql_datastore, document_dir="samples/docs")
    documents = sql_datastore.get_all_documents()
    assert len(documents) == 2
    doc = sql_datastore.get_document_by_id("1")
    assert doc.keys() == {"id", "name", "text", "tags"}
