from haystack.database.sql import SQLDocumentStore
from haystack.indexing.io import write_documents_to_db


def test_db_write_read():
    sql_datastore = SQLDocumentStore()
    write_documents_to_db(datastore=sql_datastore, document_dir="samples/docs")
    documents = sql_datastore.get_all_documents()
    assert len(documents) == 2
    assert documents[0]["text"] == 'A Doc specifically talking about haystack.\nHaystack can be used to scale QA models to large document collections.'
