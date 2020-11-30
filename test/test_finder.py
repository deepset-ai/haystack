from haystack import Finder
import pytest


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_finder_get_answers(reader, retriever_with_docs, document_store_with_docs):
    finder = Finder(reader, retriever_with_docs)
    prediction = finder.get_answers(question="Who lives in Berlin?", top_k_retriever=10,
                                    top_k_reader=3)
    assert prediction is not None
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["meta"]["meta_field"] == "test1"
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_finder_offsets(reader, retriever_with_docs, document_store_with_docs):
    finder = Finder(reader, retriever_with_docs)
    prediction = finder.get_answers(question="Who lives in Berlin?", top_k_retriever=10,
                                    top_k_reader=5)

    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    start = prediction["answers"][0]["offset_start"]
    end = prediction["answers"][0]["offset_end"]
    assert prediction["answers"][0]["context"][start:end] == prediction["answers"][0]["answer"]


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_finder_get_answers_single_result(reader, retriever_with_docs, document_store_with_docs):
    finder = Finder(reader, retriever_with_docs)
    query = "testing finder"
    prediction = finder.get_answers(question=query, top_k_retriever=1,
                                    top_k_reader=1)
    assert prediction is not None
    assert len(prediction["answers"]) == 1




