import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haystack.preview import Document
from haystack.preview.dataclasses.document import DocumentDecoder, DocumentEncoder


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc,doc_str",
    [
        (Document(text="test text"), "text: 'test text'"),
        (Document(array=np.zeros((3, 7))), "array: (3, 7)"),
        (
            Document(dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"])),
            "dataframe: (2, 2)",
        ),
        (Document(blob=bytes("hello, test string".encode("utf-8"))), "blob: 18 bytes"),
        (
            Document(
                text="test text",
                array=np.zeros((3, 7)),
                dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"]),
                blob=bytes("hello, test string".encode("utf-8")),
            ),
            "text: 'test text', array: (3, 7), dataframe: (2, 2), blob: 18 bytes",
        ),
    ],
)
def test_document_str(doc, doc_str):
    assert f"Document(id={doc.id}, mimetype: 'text/plain', {doc_str})" == str(doc)


@pytest.mark.unit
def test_init_document_same_meta_as_main_fields():
    """
    This is forbidden to prevent later issues with `Document.flatten()`
    """
    with pytest.raises(ValueError, match="score"):
        Document(text="test text", metadata={"score": "10/10"})


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
        "array": None,
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
        array=np.array([1, 2, 3]),
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some": "values", "test": 10},
        score=0.99,
        embedding=[10, 10],
    )
    dictionary = doc.to_dict()

    array = dictionary.pop("array")
    assert array.shape == doc.array.shape and (array == doc.array).all()

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
            "array": np.array([1, 2, 3]),
            "dataframe": pd.DataFrame([10, 20, 30]),
            "blob": b"some bytes",
            "mime_type": "application/pdf",
            "metadata": {"some": "values", "test": 10},
            "score": 0.99,
            "embedding": embedding,
        }
    ) == Document(
        text="test text",
        array=np.array([1, 2, 3]),
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
            "array": None,
            "dataframe": None,
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
def test_full_document_to_json(tmp_path):
    class TestClass:
        def __repr__(self):
            return "<the object>"

    doc_1 = Document(
        text="test text",
        array=np.array([1, 2, 3]),
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some object": TestClass(), "a path": tmp_path / "test.txt"},
        score=0.5,
        embedding=[1, 2, 3, 4],
    )
    assert doc_1.to_json() == json.dumps(
        {
            "id": doc_1.id,
            "text": "test text",
            "array": [1, 2, 3],
            "dataframe": '{"0":{"0":10,"1":20,"2":30}}',
            "mime_type": "application/pdf",
            "metadata": {"some object": "<the object>", "a path": str((tmp_path / "test.txt").absolute())},
            "score": 0.5,
            "embedding": [1, 2, 3, 4],
        }
    )


@pytest.mark.unit
def test_full_document_from_json(tmp_path):
    class TestClass:
        def __repr__(self):
            return "'<the object>'"

        def __eq__(self, other):
            return type(self) == type(other)

    doc = Document.from_json(
        json.dumps(
            {
                "text": "test text",
                "array": [1, 2, 3],
                "dataframe": '{"0":{"0":10,"1":20,"2":30}}',
                "mime_type": "application/pdf",
                "metadata": {"some object": "<the object>", "a path": str((tmp_path / "test.txt").absolute())},
                "score": 0.5,
                "embedding": [1, 2, 3, 4],
            }
        )
    )
    assert doc == Document(
        text="test text",
        array=np.array([1, 2, 3]),
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=None,
        mime_type="application/pdf",
        # Note the object serialization
        metadata={"some object": "<the object>", "a path": str((tmp_path / "test.txt").absolute())},
        score=0.5,
        embedding=[1, 2, 3, 4],
    )


@pytest.mark.unit
def test_to_json_custom_encoder():
    class SerializableTestClass:
        ...

    class TestEncoder(DocumentEncoder):
        def default(self, obj):
            if isinstance(obj, SerializableTestClass):
                return "<<CUSTOM ENCODING>>"
            return DocumentEncoder.default(self, obj)

    doc = Document(text="test text", metadata={"some object": SerializableTestClass()})
    doc_json = doc.to_json(indent=4, json_encoder=TestEncoder).strip()

    assert doc_json == json.dumps(
        {
            "id": doc.id,
            "text": "test text",
            "array": None,
            "dataframe": None,
            "mime_type": "text/plain",
            "metadata": {"some object": "<<CUSTOM ENCODING>>"},
            "score": None,
            "embedding": None,
        },
        indent=4,
    )


@pytest.mark.unit
def test_from_json_custom_decoder():
    class TestClass:
        def __eq__(self, other):
            return type(self) == type(other)

    class TestDecoder(DocumentDecoder):
        def __init__(self, *args, **kwargs):
            super().__init__(object_hook=self.object_hook)

        def object_hook(self, dictionary):
            if "metadata" in dictionary:
                for key, value in dictionary["metadata"].items():
                    if value == "<<CUSTOM ENCODING>>":
                        dictionary["metadata"][key] = TestClass()
            return dictionary

    doc = Document(text="test text", metadata={"some object": TestClass()})

    assert doc == Document.from_json(
        json.dumps(
            {
                "id": doc.id,
                "text": "test text",
                "array": None,
                "dataframe": None,
                "mime_type": "text/plain",
                "metadata": {"some object": "<<CUSTOM ENCODING>>"},
                "score": None,
                "embedding": None,
            }
        ),
        json_decoder=TestDecoder,
    )


@pytest.mark.unit
def test_flatten_document_no_meta():
    doc = Document(text="test text")
    assert doc.flatten() == {
        "id": doc.id,
        "text": "test text",
        "array": None,
        "dataframe": None,
        "blob": None,
        "mime_type": "text/plain",
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_flatten_document_with_flat_meta():
    doc = Document(text="test text", metadata={"some-key": "a value", "another-key": "another value!"})
    assert doc.flatten() == {
        "id": doc.id,
        "text": "test text",
        "array": None,
        "dataframe": None,
        "blob": None,
        "mime_type": "text/plain",
        "score": None,
        "embedding": None,
        "some-key": "a value",
        "another-key": "another value!",
    }


@pytest.mark.unit
def test_flatten_document_with_nested_meta():
    doc = Document(text="test text", metadata={"some-key": "a value", "nested": {"key": 10, "key2": 50}})
    assert doc.flatten() == {
        "id": doc.id,
        "text": "test text",
        "array": None,
        "dataframe": None,
        "blob": None,
        "mime_type": "text/plain",
        "score": None,
        "embedding": None,
        "some-key": "a value",
        "nested": {"key": 10, "key2": 50},
    }
