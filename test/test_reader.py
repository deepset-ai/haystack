import pytest

from haystack.reader.base import BaseReader
from haystack.database.base import Document


def test_reader_basic(reader):
    assert reader is not None
    assert isinstance(reader, BaseReader)


def test_output(reader, test_docs_xs):
    docs = []
    for d in test_docs_xs:
        doc = Document(id=d["name"], text=d["text"], meta=d["meta"])
        docs.append(doc)
    results = reader.predict(question="Who lives in Berlin?", documents=docs, top_k=5)
    assert results is not None
    assert results["question"] == "Who lives in Berlin?"
    assert results["answers"][0]["answer"] == "Carla"
    assert results["answers"][0]["offset_start"] == 11
    assert results["answers"][0]["offset_end"] == 16
    assert results["answers"][0]["probability"] <= 1
    assert results["answers"][0]["probability"] >= 0
    assert results["answers"][0]["context"] == "My name is Carla and I live in Berlin"
    assert results["answers"][0]["document_id"] == "filename1"
    assert len(results["answers"]) == 5
