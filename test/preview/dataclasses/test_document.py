from pathlib import Path

import pandas as pd
import pytest

from haystack.preview import Document
from haystack.preview.dataclasses.byte_stream import ByteStream


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc,doc_str",
    [
        (Document(content="test text"), "content: 'test text'"),
        (
            Document(dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"])),
            "dataframe: (2, 2)",
        ),
        (Document(blob=ByteStream(b"hello, test string")), "blob: 18 bytes"),
        (
            Document(
                content="test text",
                dataframe=pd.DataFrame([["John", 25], ["Martha", 34]], columns=["name", "age"]),
                blob=ByteStream(b"hello, test string"),
            ),
            "content: 'test text', dataframe: (2, 2), blob: 18 bytes",
        ),
    ],
)
def test_document_str(doc, doc_str):
    assert f"Document(id={doc.id}, {doc_str})" == str(doc)


@pytest.mark.unit
def test_init():
    doc = Document()
    assert doc.id == "d4675c57fcfe114db0b95f1da46eea3c5d6f5729c17d01fb5251ae19830a3455"
    assert doc.content == None
    assert doc.dataframe == None
    assert doc.blob == None
    assert doc.meta == {}
    assert doc.score == None
    assert doc.embedding == None


@pytest.mark.unit
def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        Document(text="")


@pytest.mark.unit
def test_init_with_parameters():
    blob_data = b"some bytes"
    doc = Document(
        content="test text",
        dataframe=pd.DataFrame([0]),
        blob=ByteStream(data=blob_data, mime_type="text/markdown"),
        meta={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
    )
    assert doc.id == "ec92455f3f4576d40031163c89b1b4210b34ea1426ee0ff68ebed86cb7ba13f8"
    assert doc.content == "test text"
    assert doc.dataframe is not None
    assert doc.dataframe.equals(pd.DataFrame([0]))
    assert doc.blob.data == blob_data
    assert doc.blob.mime_type == "text/markdown"
    assert doc.meta == {"text": "test text"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_legacy_fields():
    doc = Document(
        content="test text", content_type="text", id_hash_keys=["content"], score=0.812, embedding=[0.1, 0.2, 0.3]  # type: ignore
    )
    assert doc.id == "18fc2c114825872321cf5009827ca162f54d3be50ab9e9ffa027824b6ec223af"
    assert doc.content == "test text"
    assert doc.dataframe == None
    assert doc.blob == None
    assert doc.meta == {}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_init_with_legacy_field():
    doc = Document(
        content="test text",
        content_type="text",  # type: ignore
        id_hash_keys=["content"],  # type: ignore
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )
    assert doc.id == "a2c0321b34430cc675294611e55529fceb56140ca3202f1c59a43a8cecac1f43"
    assert doc.content == "test text"
    assert doc.dataframe == None
    assert doc.meta == {"date": "10-10-2023", "type": "article"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_basic_equality_type_mismatch():
    doc = Document(content="test text")
    assert doc != "test text"


@pytest.mark.unit
def test_basic_equality_id():
    doc1 = Document(content="test text")
    doc2 = Document(content="test text")

    assert doc1 == doc2

    doc1.id = "1234"
    doc2.id = "5678"

    assert doc1 != doc2


@pytest.mark.unit
def test_to_dict():
    doc = Document()
    assert doc.to_dict() == {
        "id": doc._create_id(),
        "content": None,
        "dataframe": None,
        "blob": None,
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_to_dict_without_flattening():
    doc = Document()
    assert doc.to_dict(flatten=False) == {
        "id": doc._create_id(),
        "content": None,
        "dataframe": None,
        "blob": None,
        "meta": {},
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_to_dict_with_custom_parameters():
    doc = Document(
        content="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=ByteStream(b"some bytes", mime_type="application/pdf"),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
    )

    assert doc.to_dict() == {
        "id": doc.id,
        "content": "test text",
        "dataframe": pd.DataFrame([10, 20, 30]).to_json(),
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf"},
        "some": "values",
        "test": 10,
        "score": 0.99,
        "embedding": [10.0, 10.0],
    }


@pytest.mark.unit
def test_to_dict_with_custom_parameters_without_flattening():
    doc = Document(
        content="test text",
        dataframe=pd.DataFrame([10, 20, 30]),
        blob=ByteStream(b"some bytes", mime_type="application/pdf"),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
    )

    assert doc.to_dict(flatten=False) == {
        "id": doc.id,
        "content": "test text",
        "dataframe": pd.DataFrame([10, 20, 30]).to_json(),
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf"},
        "meta": {"some": "values", "test": 10},
        "score": 0.99,
        "embedding": [10, 10],
    }


@pytest.mark.unit
def test_from_dict():
    assert Document.from_dict({}) == Document()


@pytest.mark.unit
def from_from_dict_with_parameters():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "dataframe": pd.DataFrame([0]).to_json(),
            "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
            "meta": {"text": "test text"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
        }
    ) == Document(
        content="test text",
        dataframe=pd.DataFrame([0]),
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        meta={"text": "test text"},
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
        content="test text", content_type="text", id_hash_keys=["content"], score=0.812, embedding=[0.1, 0.2, 0.3]  # type: ignore
    )


def test_from_dict_with_legacy_field_and_flat_meta():
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
        content_type="text",  # type: ignore
        id_hash_keys=["content"],  # type: ignore
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )


@pytest.mark.unit
def test_from_dict_with_flat_meta():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "dataframe": pd.DataFrame([0]).to_json(),
            "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        content="test text",
        dataframe=pd.DataFrame([0]),
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )


@pytest.mark.unit
def test_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'meta' parameter or flattened metadata keys"):
        Document.from_dict(
            {
                "content": "test text",
                "dataframe": pd.DataFrame([0]).to_json(),
                "blob": {"data": list(b"some bytes"), "mime_type": "text/markdown"},
                "score": 0.812,
                "meta": {"test": 10},
                "embedding": [0.1, 0.2, 0.3],
                "date": "10-10-2023",
                "type": "article",
            }
        )


@pytest.mark.unit
def test_content_type():
    assert Document(content="text").content_type == "text"
    assert Document(dataframe=pd.DataFrame([0])).content_type == "table"

    with pytest.raises(ValueError):
        _ = Document().content_type

    with pytest.raises(ValueError):
        _ = Document(content="text", dataframe=pd.DataFrame([0])).content_type
