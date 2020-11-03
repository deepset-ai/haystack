import pytest
from haystack import Finder


@pytest.mark.slow
@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch", "faiss", "memory"], indirect=True)
@pytest.mark.parametrize("retriever", ["embedding"], indirect=True)
def test_embedding_retriever(retriever, document_store):

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

    embedded = []
    for doc in documents:
        doc['embedding'] = retriever.embed([doc['meta']['question']])[0]
        embedded.append(doc)

    document_store.write_documents(embedded)

    finder = Finder(reader=None, retriever=retriever)
    prediction = finder.get_answers_via_similar_questions(question="How to test this?", top_k_retriever=1)

    assert len(prediction.get('answers', [])) == 1
