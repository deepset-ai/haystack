from haystack import Finder


def test_far_retriever_in_memory_store():
    from haystack.database.memory import InMemoryDocumentStore
    from haystack.retriever.elasticsearch import EmbeddingRetriever

    document_store = InMemoryDocumentStore()

    documents = [
        {'name': 'How to test this library?', 'text': 'By running tox in the command line!', 'meta': {'question': 'How to test this library?'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
        {'name': 'blah blah blah', 'text': 'By running tox in the command line!', 'meta': {'question': 'blah blah blah'}},
    ]

    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", gpu=False)

    embedded = []
    for doc in documents:
        doc['embedding'] = retriever.create_embedding([doc['meta']['question']])[0]
        embedded.append(doc)

    document_store.write_documents(embedded)

    finder = Finder(reader=None, retriever=retriever)
    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert prediction == {
        'question': 'How to test this?', 'answers':
            [
                {
                    'question': 'How to test this library?',
                    'answer': 'By running tox in the command line!',
                    'context': 'By running tox in the command line!',
                    'score': 0.5425953283001858,
                    'offset_start': 0,
                    'offset_end': 35,
                    'meta': {'question': 'How to test this library?'},
                    'probability': 0.7712976641500928
                 }
            ]
    }
