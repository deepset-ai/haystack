from pathlib import Path
import dataclasses
import textwrap
import json

import pytest
import pandas as pd
import numpy as np

from haystack.preview import Document
from haystack.preview.dataclasses.document import DocumentEncoder, DocumentDecoder


@pytest.mark.unit
def test_document_is_immutable():
    doc = Document(text="test text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.text = "won't work"


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
@pytest.mark.parametrize(
    "doc1_data,doc2_data",
    [
        [{"text": "test text"}, {"text": "test text", "mime_type": "text/plain"}],
        [{"text": "test text", "mime_type": "text/html"}, {"text": "test text", "mime_type": "text/plain"}],
        [{"text": "test text"}, {"text": "test text", "metadata": {"path": Path(__file__)}}],
        [
            {"text": "test text", "metadata": {"path": Path(__file__).parent}},
            {"text": "test text", "metadata": {"path": Path(__file__)}},
        ],
        [{"text": "test text"}, {"text": "test text", "score": 200}],
        [{"text": "test text", "score": 0}, {"text": "test text", "score": 200}],
        [{"text": "test text"}, {"text": "test text", "embedding": np.array([1, 2, 3])}],
        [
            {"text": "test text", "embedding": np.array([100, 222, 345])},
            {"text": "test text", "embedding": np.array([1, 2, 3])},
        ],
        [{"array": np.array(range(10))}, {"array": np.array(range(10))}],
        [{"dataframe": pd.DataFrame([1, 2, 3])}, {"dataframe": pd.DataFrame([1, 2, 3])}],
        [{"blob": b"some bytes"}, {"blob": b"some bytes"}],
    ],
)
def test_id_hash_keys_default_fields_equal_id(doc1_data, doc2_data):
    doc1 = Document.from_dict(doc1_data)
    doc2 = Document.from_dict(doc2_data)
    assert doc1.id == doc2.id


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc1_data,doc2_data",
    [
        [{"text": "test text"}, {"text": "test text "}],
        [{"array": np.array(range(10))}, {"array": np.array(range(11))}],
        [{"dataframe": pd.DataFrame([1, 2, 3])}, {"dataframe": pd.DataFrame([1, 2, 3, 4])}],
        [{"blob": b"some bytes"}, {"blob": "something else".encode()}],
    ],
)
def test_id_hash_keys_default_fields_different_ids(doc1_data, doc2_data):
    doc1 = Document.from_dict(doc1_data)
    doc2 = Document.from_dict(doc2_data)
    assert doc1.id != doc2.id


@pytest.mark.unit
def test_id_hash_keys_changes_id():
    doc1 = Document(text="test text", metadata={"some-value": "value"})
    doc2 = Document(text="test text", metadata={"some-value": "value"}, id_hash_keys=["text", "some-value"])
    assert doc1.id != doc2.id


@pytest.mark.unit
def test_id_hash_keys_field_may_be_missing(caplog):
    doc1 = Document(text="test text", id_hash_keys=["something"])
    doc2 = Document(text="test text", id_hash_keys=["something else"])
    assert doc1.id == doc2.id
    assert "is missing the following id_hash_keys: ['something']." in caplog.text
    assert "is missing the following id_hash_keys: ['something else']." in caplog.text


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

    doc1 = Document(
        text="test text", metadata={"value": np.array([0, 1, 2]), "path": Path(__file__), "obj": TestObject()}
    )
    doc2 = Document(
        text="test text", metadata={"value": np.array([0, 1, 2]), "path": Path(__file__), "obj": TestObject()}
    )
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
        "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
        id_hash_keys=["test"],
        score=0.99,
        embedding=np.zeros([10, 10]),
    )
    dictionary = doc.to_dict()

    array = dictionary.pop("array")
    assert array.shape == doc.array.shape and (array == doc.array).all()

    dataframe = dictionary.pop("dataframe")
    assert dataframe.equals(doc.dataframe)

    blob = dictionary.pop("blob")
    assert blob == doc.blob

    embedding = dictionary.pop("embedding")
    assert (embedding == doc.embedding).all()

    assert dictionary == {
        "id": doc.id,
        "text": "test text",
        "mime_type": "application/pdf",
        "metadata": {"some": "values", "test": 10},
        "id_hash_keys": ["test"],
        "score": 0.99,
    }


@pytest.mark.unit
def test_document_with_most_attributes_from_dict():
    embedding = np.zeros([10, 10])
    assert Document.from_dict(
        {
            "text": "test text",
            "array": np.array([1, 2, 3]),
            "dataframe": pd.DataFrame([10, 20, 30]),
            "blob": b"some bytes",
            "mime_type": "application/pdf",
            "metadata": {"some": "values", "test": 10},
            "id_hash_keys": ["test"],
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
        id_hash_keys=["test"],
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
            "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
        id_hash_keys=["test"],
        score=0.5,
        embedding=np.array([1, 2, 3, 4]),
    )
    assert doc_1.to_json() == json.dumps(
        {
            "id": doc_1.id,
            "text": "test text",
            "array": [1, 2, 3],
            "dataframe": '{"0":{"0":10,"1":20,"2":30}}',
            "mime_type": "application/pdf",
            "metadata": {"some object": "<the object>", "a path": str((tmp_path / "test.txt").absolute())},
            "id_hash_keys": ["test"],
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
                "id_hash_keys": ["test"],
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
        id_hash_keys=["test"],
        score=0.5,
        embedding=np.array([1, 2, 3, 4]),
    )


@pytest.mark.unit
def test_to_json_custom_encoder(tmp_path):
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
            "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
                "text": "test text",
                "array": None,
                "dataframe": None,
                "mime_type": "text/plain",
                "metadata": {"some object": "<<CUSTOM ENCODING>>"},
                "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
        "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
        "id_hash_keys": ["text", "array", "dataframe", "blob"],
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
        "id_hash_keys": ["text", "array", "dataframe", "blob"],
        "score": None,
        "embedding": None,
        "some-key": "a value",
        "nested": {"key": 10, "key2": 50},
    }
