

def test_tfidf_retriever(inmemory_document_store, tfidf_retriever):

    test_docs = [
        {"id": "26f84672c6d7aaeb8e2cd53e9c62d62d", "name": "testing the finder 1", "text": "godzilla says hello"},
        {"name": "testing the finder 2", "text": "optimus prime says bye"},
        {"name": "testing the finder 3", "text": "alien says arghh"}
    ]

    inmemory_document_store.write_documents(test_docs)

    tfidf_retriever.fit()
    doc = tfidf_retriever.retrieve("godzilla", top_k=1)[0]
    assert doc.id == "26f84672c6d7aaeb8e2cd53e9c62d62d"
    assert doc.text == 'godzilla says hello'
    assert doc.meta == {"name": "testing the finder 1"}
