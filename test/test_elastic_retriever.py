from haystack.retriever.sparse import ElasticsearchRetriever
import pytest


@pytest.mark.parametrize("document_store_with_docs", [("elasticsearch")], indirect=True)
def test_elasticsearch_retrieval(document_store_with_docs):
    retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    res = retriever.retrieve(query="Who lives in Berlin?")
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 3
    assert res[0].meta["name"] == "filename1"

@pytest.mark.parametrize("document_store_with_docs", [("elasticsearch")], indirect=True)
def test_elasticsearch_retrieval_filters(document_store_with_docs):
    retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    res = retriever.retrieve(query="Who lives in Berlin?", filters={"name": ["filename1"]})
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 1
    assert res[0].meta["name"] == "filename1"

    res = retriever.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field": ["not_existing_value"]})
    assert len(res) == 0

    res = retriever.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "not_existing_field": ["not_existing_value"]})
    assert len(res) == 0

    retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    res = retriever.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field": ["test1","test2"]})
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 1
    assert res[0].meta["name"] == "filename1"

    retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    res = retriever.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field":["test2"]})
    assert len(res) == 0
