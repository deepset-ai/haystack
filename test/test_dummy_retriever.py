from haystack.database.base import Document


def test_dummy_retriever():
    from haystack.retriever.sparse import ElasticsearchFilterOnlyRetriever

    test_docs = [
        {"name": "test1", "text": "godzilla says hello"},
        {"name": "test2", "text": "optimus prime says bye"},
        {"name": "test3", "text": "alien says arghh"}
    ]

    from haystack.database.elasticsearch import ElasticsearchDocumentStore
    document_store = ElasticsearchDocumentStore()
    document_store.write_documents(test_docs)

    retriever = ElasticsearchFilterOnlyRetriever(document_store)
    result = retriever.retrieve(query="godzilla", filters={"name": ["test1"]}, top_k=1)
    assert type(result[0]) == Document
    assert result[0].text == "godzilla says hello"
    assert result[0].meta["name"] == "test1"