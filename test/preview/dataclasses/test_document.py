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
def test_init():
    doc = Document()
    assert doc.id == "eaefbcfb6d4274ef83b7b4726d5df854060b6079d12bac65e8ed3feb99d9f69e"
    assert doc.text == None
    assert doc.dataframe == None
    assert doc.blob == None
    assert doc.mime_type == "text/plain"
    assert doc.metadata == {}
    assert doc.metadata == {}
    assert doc.score == None
    assert doc.embedding == None


@pytest.mark.unit
def test_init_with_parameters():
    blob = b"some bytes"
    doc = Document(
        text="test text",
        dataframe=pd.DataFrame([0]),
        blob=blob,
        mime_type="text/markdown",
        metadata={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
    )
    assert doc.id == "ec92455f3f4576d40031163c89b1b4210b34ea1426ee0ff68ebed86cb7ba13f8"
    assert doc.text == "test text"
    assert doc.dataframe.equals(pd.DataFrame([0]))
    assert doc.blob == blob
    assert doc.mime_type == "text/markdown"
    assert doc.metadata == {"text": "test text"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_legacy_fields():
    doc = Document(
        content="test text", content_type="text", id_hash_keys=["content"], score=0.812, embedding=[0.1, 0.2, 0.3]
    )
    assert doc.id == "c7f3af4f4010b88e830e4dd4f93060baeea747518642293db6325e6563a1ce37"
    assert doc.text == "test text"
    assert doc.dataframe == None
    assert doc.blob == None
    assert doc.mime_type == "text/plain"
    assert doc.metadata == {}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_legacy_field_and_flat_metadata():
    doc = Document(
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        date="10-10-2023",
        type="article",
    )
    assert doc.id == "523cc14d7d8ce5e2fc69940969c40c5860a621e17f1c61eaa2655356519ac36d"
    assert doc.text == "test text"
    assert doc.dataframe == None
    assert doc.blob == None
    assert doc.mime_type == "text/plain"
    assert doc.metadata == {"date": "10-10-2023", "type": "article"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_flat_metadata():
    blob = b"some bytes"
    doc = Document(
        text="test text",
        dataframe=pd.DataFrame([0]),
        blob=blob,
        mime_type="text/markdown",
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        date="10-10-2023",
        type="article",
    )
    assert doc.id == "c6212ad7bb513c572367e11dd12fd671911a1a5499e3d31e4fe3bda7e87c0641"
    assert doc.text == "test text"
    assert doc.dataframe.equals(pd.DataFrame([0]))
    assert doc.blob == blob
    assert doc.mime_type == "text/markdown"
    assert doc.metadata == {"date": "10-10-2023", "type": "article"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_flat_and_non_flat_metadata():
    with pytest.raises(TypeError):
        Document(
            text="test text",
            dataframe=pd.DataFrame([0]),
            blob=b"some bytes",
            mime_type="text/markdown",
            score=0.812,
            metadata={"test": 10},
            embedding=[0.1, 0.2, 0.3],
            date="10-10-2023",
            type="article",
        )


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
def test_to_dict():
    doc = Document()
    assert doc.to_dict() == {
        "id": doc._create_id(),
        "text": None,
        "dataframe": None,
        "blob": None,
        "mime_type": "text/plain",
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_to_dict_without_flattening():
    doc = Document()
    assert doc.to_dict(flatten=False) == {
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
def test_to_dict_with_custom_parameters():
    doc = Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some": "values", "test": 10},
        score=0.99,
        embedding=[10, 10],
    )

    assert doc.to_dict() == {
        "id": doc.id,
        "text": "test text",
        "dataframe": pd.DataFrame([10, 20, 30]).to_json(),
        "blob": list(doc.blob),
        "mime_type": "application/pdf",
        "some": "values",
        "test": 10,
        "score": 0.99,
        "embedding": [10, 10],
    }


@pytest.mark.unit
def test_to_dict_with_custom_parameters_without_flattening():
    doc = Document(
        text="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=b"some bytes",
        mime_type="application/pdf",
        metadata={"some": "values", "test": 10},
        score=0.99,
        embedding=[10, 10],
    )

    assert doc.to_dict(flatten=False) == {
        "id": doc.id,
        "text": "test text",
        "dataframe": pd.DataFrame([10, 20, 30]).to_json(),
        "blob": list(doc.blob),
        "mime_type": "application/pdf",
        "metadata": {"some": "values", "test": 10},
        "score": 0.99,
        "embedding": [10, 10],
    }


@pytest.mark.unit
def test_from_dict():
    Document.from_dict({}) == Document()


@pytest.mark.unit
def from_from_dict_with_parameters():
    blob = b"some bytes"
    assert Document.from_dict(
        {
            "text": "test text",
            "dataframe": pd.DataFrame([0]).to_json(),
            "blob": blob,
            "mime_type": "text/markdown",
            "metadata": {"text": "test text"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
        }
    ) == Document(
        text="test text",
        dataframe=pd.DataFrame([0]),
        blob=blob,
        mime_type="text/markdown",
        metadata={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.mark.unit
def test_from_dict_with_legacy_fields():
    assert Document.from_dict(
        {
            "content": "test text",
            "content_type": "text",
            "id_hash_keys": ["content"],
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
        }
    ) == Document(
        content="test text", content_type="text", id_hash_keys=["content"], score=0.812, embedding=[0.1, 0.2, 0.3]
    )


def test_from_dict_with_legacy_field_and_flat_metadata():
    assert Document.from_dict(
        {
            "content": "test text",
            "content_type": "text",
            "id_hash_keys": ["content"],
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        date="10-10-2023",
        type="article",
    )


@pytest.mark.unit
def test_from_dict_with_flat_metadata():
    blob = b"some bytes"
    assert Document.from_dict(
        {
            "text": "test text",
            "dataframe": pd.DataFrame([0]).to_json(),
            "blob": blob,
            "mime_type": "text/markdown",
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        text="test text",
        dataframe=pd.DataFrame([0]),
        blob=blob,
        mime_type="text/markdown",
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        metadata={"date": "10-10-2023", "type": "article"},
    )


@pytest.mark.unit
def test_from_dict_with_flat_and_non_flat_metadata():
    with pytest.raises(TypeError):
        Document.from_dict(
            {
                "text": "test text",
                "dataframe": pd.DataFrame([0]).to_json(),
                "blob": b"some bytes",
                "mime_type": "text/markdown",
                "score": 0.812,
                "metadata": {"test": 10},
                "embedding": [0.1, 0.2, 0.3],
                "date": "10-10-2023",
                "type": "article",
            }
        )


@pytest.mark.unit
def test_content_type():
    assert Document(text="text").content_type == "text"
    assert Document(dataframe=pd.DataFrame([0])).content_type == "table"

    with pytest.raises(ValueError):
        Document().content_type

    with pytest.raises(ValueError):
        Document(text="text", dataframe=pd.DataFrame([0])).content_type
