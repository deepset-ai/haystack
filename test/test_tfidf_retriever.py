from haystack.database.base import Document


def test_tfidf_retriever():
    from haystack.retriever.tfidf import TfidfRetriever

    test_docs = [
        {"name": "testing the finder 1", "text": "godzilla says hello"},
        {"name": "testing the finder 2", "text": "optimus prime says bye"},
        {"name": "testing the finder 3", "text": "alien says arghh"}
    ]

    from haystack.database.memory import InMemoryDocumentStore
    document_store = InMemoryDocumentStore()
    document_store.write_documents(test_docs)

    retriever = TfidfRetriever(document_store)
    retriever.fit()
    assert retriever.retrieve("godzilla", top_k=1) == [Document(id='0', text='godzilla says hello', external_source_id=None, question=None, query_score=None, meta=None)]