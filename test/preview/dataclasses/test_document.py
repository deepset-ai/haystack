import json
from pathlib import Path

import pandas as pd
import pytest

from haystack.preview import Document


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc,doc_str",
    [
        (Document(text="test text"), "text: 'test text'"),
        (
            Document(dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"])),
            "dataframe: (2, 2)",
        ),
        (Document(blob=bytes("hello, test string".encode("utf-8"))), "blob: 18 bytes"),
        (
            Document(
                text="test text",
                dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"]),
                blob=bytes("hello, test string".encode("utf-8")),
            ),
            "text: 'test text', dataframe: (2, 2), blob: 18 bytes",
        ),
    ],
)
def test_document_str(doc, doc_str):
    assert f"Document(id={doc.id}, mimetype: 'text/plain', {doc_str})" == str(doc)


@pytest.mark.unit
def test_basic_equality_type_mismatch():
    doc = Document(text="test text")
    assert doc != "test text"


@pytest.mark.unit
def test_basic_equality_id():
    doc1 = Document(text="test text")
    doc2 = Document(text="test text")

    assert doc1 == doc2

    object.__setattr__(doc1, "id", "1234")
    object.__setattr__(doc2, "id", "5678")

    assert doc1 != doc2


@pytest.mark.unit
def test_equality_with_metadata_with_objects():
    class TestObject:
        def __eq__(self, other):
            if type(self) == type(other):
                return True

    foo = TestObject()
    doc1 = Document(text="test text", metadata={"value": [0, 1, 2], "path": Path("."), "obj": foo})
    doc2 = Document(text="test text", metadata={"value": [0, 1, 2], "path": Path("."), "obj": foo})
    assert doc1 == doc2


@pytest.mark.unit
def test_empty_document_to_dict():
    doc = Document()
    assert doc.to_dict() == {
        "id": doc._create_id(),
        "text": None,
        "dataframe": None,
        "blob": None,
        "mime_type": "text/plain",
        "metadata": {},
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_empty_document_from_dict():
    assert Document.from_dict({}) == Document()


@pytest.mark.unit
def test_full_document_to_dict():
    doc = Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some": "values", "test": 10},
        score=0.99,
        embedding=[10, 10],
    )
    dictionary = doc.to_dict()

    dataframe = dictionary.pop("dataframe")
    assert dataframe.equals(doc.dataframe)

    blob = dictionary.pop("blob")
    assert blob == doc.blob

    embedding = dictionary.pop("embedding")
    assert embedding == doc.embedding

    assert dictionary == {
        "id": doc.id,
        "text": "test text",
        "mime_type": "application/pdf",
        "metadata": {"some": "values", "test": 10},
        "score": 0.99,
    }


@pytest.mark.unit
def test_document_with_most_attributes_from_dict():
    embedding = [10, 10]
    assert Document.from_dict(
        {
            "text": "test text",
            "dataframe": pd.DataFrame([10, 20, 30]),
            "blob": b"some bytes",
            "mime_type": "application/pdf",
            "metadata": {"some": "values", "test": 10},
            "score": 0.99,
            "embedding": embedding,
        }
    ) == Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some": "values", "test": 10},
        score=0.99,
        embedding=embedding,
    )


@pytest.mark.unit
def test_empty_document_to_json():
    doc = Document()
    assert doc.to_json() == json.dumps(
        {
            "id": doc.id,
            "text": None,
            "dataframe": None,
            "blob": None,
            "mime_type": "text/plain",
            "metadata": {},
            "score": None,
            "embedding": None,
        }
    )


@pytest.mark.unit
def test_empty_document_from_json():
    assert Document.from_json("{}") == Document()


@pytest.mark.unit
def test_full_document_to_json():
    doc_1 = Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"create_date": -14186580},
        score=0.5,
        embedding=[1, 2, 3, 4],
    )
    assert doc_1.to_json() == json.dumps(
        {
            "id": doc_1.id,
            "text": "test text",
            "dataframe": '{"0":{"0":10,"1":20,"2":30}}',
            "blob": list(b"some bytes"),
            "mime_type": "application/pdf",
            "metadata": {"create_date": -14186580},
            "score": 0.5,
            "embedding": [1, 2, 3, 4],
        }
    )


@pytest.mark.unit
def test_full_document_from_json():
    doc = Document.from_json(
        json.dumps(
            {
                "text": "test text",
                "dataframe": '{"0":{"0":10,"1":20,"2":30}}',
                "mime_type": "application/pdf",
                "metadata": {"create_date": -14186580},
                "score": 0.5,
                "embedding": [1, 2, 3, 4],
            }
        )
    )
    assert doc == Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=None,
        mime_type="application/pdf",
        metadata={"create_date": -14186580},
        score=0.5,
        embedding=[1, 2, 3, 4],
    )
