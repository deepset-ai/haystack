from haystack import Finder


def test_faq_retriever_in_memory_store():

    from haystack.database.memory import InMemoryDocumentStore
    from haystack.retriever.dense import EmbeddingRetriever

    document_store = InMemoryDocumentStore(embedding_field="embedding")

    documents = [
        {'text': 'By running tox in the command line!', 'meta': {'name': 'How to test this library?', 'question': 'How to test this library?'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
        {'text': 'By running tox in the command line!', 'meta': {'name': 'blah blah blah', 'question': 'blah blah blah'}},
    ]

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)

    embedded = []
    for doc in documents:
        doc['embedding'] = retriever.embed([doc['meta']['question']])[0]
        embedded.append(doc)

    document_store.write_documents(embedded)

    finder = Finder(reader=None, retriever=retriever)
    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1
