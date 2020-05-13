from haystack import Finder
from haystack.reader.transformers import TransformersReader
from haystack.retriever.tfidf import TfidfRetriever


def test_finder_get_answers_with_in_memory_store():
    test_docs = [
        {"name": "testing the finder 1", "text": "testing the finder with pyhton unit test 1", 'meta': {'url': 'url'}},
        {"name": "testing the finder 2", "text": "testing the finder with pyhton unit test 2", 'meta': {'url': 'url'}},
        {"name": "testing the finder 3", "text": "testing the finder with pyhton unit test 3", 'meta': {'url': 'url'}}
    ]

    from haystack.database.memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents(test_docs, tags=[{'has_url': ["true"] for _ in range(0, len(test_docs))}])

    retriever = TfidfRetriever(document_store=document_store)
    reader = TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                tokenizer="distilbert-base-uncased", use_gpu=-1)
    finder = Finder(reader, retriever)
    prediction = finder.get_answers(question="testing finder", top_k_retriever=10,
                                    top_k_reader=5)
    assert prediction is not None


def test_memory_store_get_by_tags():
    test_docs = [
        {"name": "testing the finder 1", "text": "testing the finder with pyhton unit test 1", 'meta': {'url': 'url'}},
        {"name": "testing the finder 2", "text": "testing the finder with pyhton unit test 2", 'meta': {'url': None}},
        {"name": "testing the finder 3", "text": "testing the finder with pyhton unit test 3", 'meta': {'url': 'url'}}
    ]

    from haystack.database.memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents(test_docs, tags=[None, {'has_url': 'false'}, {'has_url': 'true'}])

    docs = document_store.get_document_ids_by_tags({'has_url': 'false'})

    assert docs == [{'name': 'testing the finder 2', 'text': 'testing the finder with pyhton unit test 2', 'meta': {'url': None}}]


def test_memory_store_get_by_tag_lists():
    test_docs = [
        {"name": "testing the finder 1", "text": "testing the finder with pyhton unit test 1", 'meta': {'url': 'url'}},
        {"name": "testing the finder 2", "text": "testing the finder with pyhton unit test 2", 'meta': {'url': None}},
        {"name": "testing the finder 3", "text": "testing the finder with pyhton unit test 3", 'meta': {'url': 'url'}}
    ]

    from haystack.database.memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents(test_docs, tags=[{'tag2': ["1"]}, {'tag1': ['1']}, {'tag2': ["1", "2"]}])

    docs = document_store.get_document_ids_by_tags({'tag2': ["1"]})

    print(docs)

    assert docs == [{'name': 'testing the finder 1', 'text': 'testing the finder with pyhton unit test 1', 'meta': {'url': 'url'}}]
