import pytest
from elasticsearch import Elasticsearch

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_with_docs", [("elasticsearch")], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch"], indirect=True)
def test_elasticsearch_retrieval(retriever_with_docs, document_store_with_docs):
    res = retriever_with_docs.retrieve(query="Who lives in Berlin?")
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 3
    assert res[0].meta["name"] == "filename1"


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store_with_docs", [("elasticsearch")], indirect=True)
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch"], indirect=True)
def test_elasticsearch_retrieval_filters(retriever_with_docs, document_store_with_docs):
    res = retriever_with_docs.retrieve(query="Who lives in Berlin?", filters={"name": ["filename1"]})
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 1
    assert res[0].meta["name"] == "filename1"

    res = retriever_with_docs.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field": ["not_existing_value"]})
    assert len(res) == 0

    res = retriever_with_docs.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "not_existing_field": ["not_existing_value"]})
    assert len(res) == 0

    res = retriever_with_docs.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field": ["test1","test2"]})
    assert res[0].text == "My name is Carla and I live in Berlin"
    assert len(res) == 1
    assert res[0].meta["name"] == "filename1"

    res = retriever_with_docs.retrieve(query="Who lives in Berlin?", filters={"name":["filename1"], "meta_field":["test2"]})
    assert len(res) == 0


@pytest.mark.elasticsearch
def test_elasticsearch_custom_query(elasticsearch_fixture):
    client = Elasticsearch()
    client.indices.delete(index='haystack_test_custom', ignore=[404])
    document_store = ElasticsearchDocumentStore(index="haystack_test_custom", text_field="custom_text_field",
                                                embedding_field="custom_embedding_field")
    documents = [
        {"text": "test_1", "meta": {"year": "2019"}},
        {"text": "test_2", "meta": {"year": "2020"}},
        {"text": "test_3", "meta": {"year": "2021"}},
        {"text": "test_4", "meta": {"year": "2021"}},
        {"text": "test_5", "meta": {"year": "2021"}},
    ]
    document_store.write_documents(documents)

    # test custom "terms" query
    retriever = ElasticsearchRetriever(
        document_store=document_store,
        custom_query="""
            {
                "size": 10, 
                "query": {
                    "bool": {
                        "should": [{
                            "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["text"]}}],
                            "filter": [{"terms": {"year": ${years}}}]}}}"""
    )
    results = retriever.run(query="test", filters={"years": ["2020", "2021"]})[0]["documents"]
    assert len(results) == 4

    # test custom "term" query
    retriever = ElasticsearchRetriever(
        document_store=document_store,
        custom_query="""
                {
                    "size": 10, 
                    "query": {
                        "bool": {
                            "should": [{
                                "multi_match": {"query": ${query}, "type": "most_fields", "fields": ["text"]}}],
                                "filter": [{"term": {"year": ${years}}}]}}}"""
    )
    results = retriever.run(query="test", filters={"years": "2021"})[0]["documents"]
    assert len(results) == 3
