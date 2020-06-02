from time import sleep

from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.database.sql import SQLDocumentStore
from haystack.indexing.utils import convert_files_to_documents


def test_sql_write_read():
    sql_document_store = SQLDocumentStore()
    documents = convert_files_to_documents(dir_path="samples/docs")
    sql_document_store.write_documents(documents)
    documents = sql_document_store.get_all_documents()
    assert len(documents) == 2
    doc = sql_document_store.get_document_by_id("1")
    assert doc.id
    assert doc.text


def test_elasticsearch_write_read(elasticsearch_fixture):
    document_store = ElasticsearchDocumentStore()
    documents = convert_files_to_documents(dir_path="samples/docs")
    document_store.write_documents(documents)
    sleep(2)  # wait for documents to be available for query
    documents = document_store.get_all_documents()
    assert len(documents) == 2
    assert documents[0].id
    assert documents[0].text
