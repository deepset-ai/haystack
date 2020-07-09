from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import TfidfRetriever
import os

def test_finder_get_answers():
    test_docs = [
        {"name": "filename1", "text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1"}},
        {"name": "filename2", "text": "My name is Paul and I live in New York", "meta": {"meta_field": "test2"}},
        {"name": "filename3", "text": "My name is Christelle and I live in Paris", "meta": {"meta_field": "test3"}}
    ]
    if os.path.exists("qa_test.db"):
        os.remove("qa_test.db")

    document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
    document_store.write_documents(test_docs)
    retriever = TfidfRetriever(document_store=document_store)
    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="Who lives in Berlin?", top_k_retriever=10,
                                    top_k_reader=5)
    assert prediction is not None
    assert prediction["question"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
    assert prediction["answers"][0]["offset_start"] == 11
    assert prediction["answers"][0]["offset_end"] == 16
    assert prediction["answers"][0]["probability"] <= 1
    assert prediction["answers"][0]["probability"] >= 0
    assert prediction["answers"][0]["meta"]["meta_field"] == "test1"
    assert prediction["answers"][0]["context"] == "My name is Carla and I live in Berlin"
    assert prediction["answers"][0]["document_id"] == "0"

    assert len(prediction["answers"]) == 5


def test_finder_get_answers_single_result():
    test_docs = [
        {"name": "filename1", "text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1"}},
        {"name": "filename2", "text": "My name is Paul and I live in New York", "meta": {"meta_field": "test2"}},
        {"name": "filename3", "text": "My name is Christelle and I live in Paris", "meta": {"meta_field": "test3"}}
    ]

    if os.path.exists("qa_test.db"):
        os.remove("qa_test.db")

    document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
    document_store.write_documents(test_docs)
    retriever = TfidfRetriever(document_store=document_store)
    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="testing finder", top_k_retriever=1,
                                    top_k_reader=1)
    assert prediction is not None
    assert len(prediction["answers"]) == 1




