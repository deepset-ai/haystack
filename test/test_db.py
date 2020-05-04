from time import sleep

from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.database.sql import SQLDocumentStore
from haystack.indexing.io import write_documents_to_db


def test_sql_write_read():
    sql_document_store = SQLDocumentStore()
    write_documents_to_db(document_store=sql_document_store, document_dir="samples/docs")
    documents = sql_document_store.get_all_documents()
    assert len(documents) == 2
    doc = sql_document_store.get_document_by_id("1")
    assert doc.id
    assert doc.text


def test_elasticsearch_write_read(elasticsearch_fixture):
    document_store = ElasticsearchDocumentStore()
    write_documents_to_db(document_store=document_store, document_dir="samples/docs")
    sleep(2)  # wait for documents to be available for query
    documents = document_store.get_all_documents()
    assert len(documents) == 2
    assert documents[0].id
    assert documents[0].text
