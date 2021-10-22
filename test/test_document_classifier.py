import pytest

from haystack import Document
from haystack.document_classifier.base import BaseDocumentClassifier


@pytest.mark.slow
def test_document_classifier(document_classifier):
    assert isinstance(document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it."""*700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(
            content="""That's bad. I don't like it.""",
            meta={"name": "1"},
            id="2",
        ),
    ]
    results = document_classifier.predict(documents=docs)
    expected_labels = ["joy", "sadness"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]


@pytest.mark.slow
def test_zero_shot_document_classifier(zero_shot_document_classifier):
    assert isinstance(zero_shot_document_classifier, BaseDocumentClassifier)

    docs = [
        Document(
            content="""That's good. I like it."""*700,  # extra long text to check truncation
            meta={"name": "0"},
            id="1",
        ),
        Document(
            content="""That's bad. I don't like it.""",
            meta={"name": "1"},
            id="2",
        ),
    ]
    results = zero_shot_document_classifier.predict(documents=docs)
    expected_labels = ["positive", "negative"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]
