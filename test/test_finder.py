from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.reader.transformers import TransformersReader
from haystack.retriever.tfidf import TfidfRetriever


def test_finder_get_answers():
    test_docs = [
        {"name": "testing the finder 1", "text": "testing the finder with pyhton unit test 1"},
        {"name": "testing the finder 2", "text": "testing the finder with pyhton unit test 2"},
        {"name": "testing the finder 3", "text": "testing the finder with pyhton unit test 3"}
    ]

    document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
    document_store.write_documents(test_docs)
    retriever = TfidfRetriever(document_store=document_store)
    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="testing finder", top_k_retriever=10,
                                    top_k_reader=5)
    assert prediction is not None


def test_finder_get_answers_single_result():
    test_docs = [
        {"name": "testing the finder 1", "text": "testing the finder with pyhton unit test 1"},
        {"name": "testing the finder 2", "text": "testing the finder with pyhton unit test 2"},
        {"name": "testing the finder 3", "text": "testing the finder with pyhton unit test 3"}
    ]

    document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
    document_store.write_documents(test_docs)
    retriever = TfidfRetriever(document_store=document_store)
    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="testing finder", top_k_retriever=1,
                                    top_k_reader=1)
    assert prediction is not None
