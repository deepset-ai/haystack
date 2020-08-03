from haystack import Finder
from haystack.retriever.sparse import TfidfRetriever
import pytest


def test_finder_get_answers(reader, document_store_with_docs):
    retriever = TfidfRetriever(document_store=document_store_with_docs)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="Who lives in Berlin?", top_k_retriever=10,
                                    top_k_reader=3)
    assert prediction is not None
    assert prediction["question"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["meta"]["meta_field"] == "test1"
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"

    assert len(prediction["answers"]) == 3


def test_finder_offsets(reader, document_store_with_docs):
    retriever = TfidfRetriever(document_store=document_store_with_docs)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="Who lives in Berlin?", top_k_retriever=10,
                                    top_k_reader=5)

    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    start = prediction["answers"][0]["offset_start"]
    end = prediction["answers"][0]["offset_end"]
    assert prediction["answers"][0]["context"][start:end] == prediction["answers"][0]["answer"]


def test_finder_get_answers_single_result(reader, document_store_with_docs):
    retriever = TfidfRetriever(document_store=document_store_with_docs)
    finder = Finder(reader, retriever)
    query = "testing finder"
    prediction = finder.get_answers(question=query, top_k_retriever=1,
                                    top_k_reader=1)
    assert prediction is not None
    assert len(prediction["answers"]) == 1




