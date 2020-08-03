from haystack.database.base import Document
from uuid import UUID

def test_tfidf_retriever():
    from haystack.retriever.sparse import TfidfRetriever

    test_docs = [
        {"id": "26f84672c6d7aaeb8e2cd53e9c62d62d", "name": "testing the finder 1", "text": "godzilla says hello"},
        {"name": "testing the finder 2", "text": "optimus prime says bye"},
        {"name": "testing the finder 3", "text": "alien says arghh"}
    ]

    from haystack.database.memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents(test_docs)

    retriever = TfidfRetriever(document_store)
    retriever.fit()
    doc = retriever.retrieve("godzilla", top_k=1)[0]
    assert doc.id == UUID("26f84672c6d7aaeb8e2cd53e9c62d62d", version=4)
    assert doc.text == 'godzilla says hello'
    assert doc.meta == {"name": "testing the finder 1"}
