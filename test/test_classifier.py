from haystack import Document
from haystack.classifier.base import BaseClassifier


def test_classifier(classifier):
    assert isinstance(classifier, BaseClassifier)

    query = "not used at the moment"
    docs = [
        Document(
            text="""Fragen und Antworten - Bitte auf Themen beschränken	welche einen Bezug zur Bahn aufweisen. Persönliche Unterhaltungen bitte per PN führen. Links bitte mit kurzer Erklärung zum verlinkten Inhalt versehen""",
            meta={"name": "0"},
            id="1",
        ),
        Document(
            text="""Ich liebe es wenn die Bahn selbstverschuldete unnötig lange Aufenthaltszeiten durch Verspätung wieder rausfährt.""",
            meta={"name": "1"},
            id="2",
        ),
    ]
    results = classifier.predict(query=query, documents=docs)
    expected_labels = ["neutral", "negative"]
    for i, doc in enumerate(results):
        assert doc.to_dict()["meta"]["classification"]["label"] == expected_labels[i]
