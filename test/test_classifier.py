from haystack import Document
from haystack.classifier.base import BaseClassifier


def test_classifier(classifier):
    assert isinstance(classifier, BaseClassifier)

    docs = [
        Document(
            text="""That's good. I like it.""",
            meta={"name": "0"},
            id="1",
        ),
        Document(
            text="""That's bad. I don't like it.""",
            meta={"name": "1"},
            id="2",
        ),
    ]
    results = classifier.predict(documents=docs)
    expected_labels = ["positive", "negative"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]