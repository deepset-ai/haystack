import pytest


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
