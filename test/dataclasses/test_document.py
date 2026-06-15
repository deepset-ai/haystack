# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from copy import deepcopy
from dataclasses import replace

import pytest

from haystack import Document
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.sparse_embedding import SparseEmbedding


@pytest.mark.parametrize(
    "doc,doc_str",
    [
        (Document(content="test text"), "content: 'test text'"),
        (Document(blob=ByteStream(b"hello, test string")), "blob: 18 bytes"),
        (Document(content="test text", blob=ByteStream(b"hello, test string")), "content: 'test text', blob: 18 bytes"),
    ],
)
def test_document_str(doc, doc_str):
    assert f"Document(id={doc.id}, {doc_str})" == str(doc)


def test_init():
    doc = Document()
    assert doc.id == "d4675c57fcfe114db0b95f1da46eea3c5d6f5729c17d01fb5251ae19830a3455"
    assert doc.content is None
    assert doc.blob is None
    assert doc.meta == {}
    assert doc.score is None
    assert doc.embedding is None
    assert doc.sparse_embedding is None


def test_init_with_wrong_parameters():
    with pytest.raises(TypeError):
        Document(text="")  # type: ignore[call-arg]


def test_init_with_parameters():
    blob_data = b"some bytes"
    sparse_embedding = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
    doc = Document(
        content="test text",
        blob=ByteStream(data=blob_data, mime_type="text/markdown"),
        meta={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=sparse_embedding,
    )
    assert doc.id == "1aa43af57c1dbc317241bf55d3067049f334d3b458d95dc72f71a7111f6c1a56"
    assert doc.content == "test text"
    assert doc.blob is not None
    assert doc.blob.data == blob_data
    assert doc.blob.mime_type == "text/markdown"
    assert doc.meta == {"text": "test text"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding == sparse_embedding


def test_init_with_legacy_fields():
    doc = Document(
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        dataframe="placeholder",
        score=0.812,
        embedding=[0.1, 0.2, 0.3],  # type: ignore
    )
    assert doc.id == "18fc2c114825872321cf5009827ca162f54d3be50ab9e9ffa027824b6ec223af"
    assert doc.content == "test text"
    assert doc.blob is None
    assert doc.meta == {}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding is None

    assert doc.content_type == "text"  # this is a property now

    assert not hasattr(doc, "id_hash_keys")
    assert not hasattr(doc, "dataframe")


def test_init_with_legacy_field():
    doc = Document(
        content="test text",
        content_type="text",  # type: ignore
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )
    assert doc.id == "a2c0321b34430cc675294611e55529fceb56140ca3202f1c59a43a8cecac1f43"
    assert doc.content == "test text"
    assert doc.meta == {"date": "10-10-2023", "type": "article"}
    assert doc.score == 0.812
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.sparse_embedding is None

    assert doc.content_type == "text"  # this is a property now
    assert not hasattr(doc, "id_hash_keys")


def test_basic_equality_type_mismatch():
    doc = Document(content="test text")
    assert doc != "test text"


def test_basic_equality_id():
    doc1 = Document(content="test text")
    doc2 = Document(content="test text")

    assert doc1 == doc2

    doc1 = replace(doc1, id="1234")
    doc2 = replace(doc2, id="5678")

    assert doc1 != doc2


def test_to_dict():
    doc = Document()
    assert doc.to_dict() == {
        "id": doc._create_id(),
        "content": None,
        "blob": None,
        "score": None,
        "embedding": None,
        "sparse_embedding": None,
    }


def test_to_dict_without_flattening():
    doc = Document()
    assert doc.to_dict(flatten=False) == {
        "id": doc._create_id(),
        "content": None,
        "blob": None,
        "meta": {},
        "score": None,
        "embedding": None,
        "sparse_embedding": None,
    }


def test_to_dict_with_custom_parameters():
    doc = Document(
        content="test text",
        blob=ByteStream(b"some bytes", mime_type="application/pdf", meta={"foo": "bar"}),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )

    assert doc.to_dict() == {
        "id": doc.id,
        "content": "test text",
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf", "meta": {"foo": "bar"}},
        "some": "values",
        "test": 10,
        "score": 0.99,
        "embedding": [10.0, 10.0],
        "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
    }


def test_to_dict_with_custom_parameters_without_flattening():
    doc = Document(
        content="test text",
        blob=ByteStream(b"some bytes", mime_type="application/pdf"),
        meta={"some": "values", "test": 10},
        score=0.99,
        embedding=[10.0, 10.0],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )

    assert doc.to_dict(flatten=False) == {
        "id": doc.id,
        "content": "test text",
        "blob": {"data": list(b"some bytes"), "mime_type": "application/pdf", "meta": {}},
        "meta": {"some": "values", "test": 10},
        "score": 0.99,
        "embedding": [10.0, 10.0],
        "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
    }


def test_to_dict_field_precedence():
    """
    Test for Document.to_dict() with flatten=True.

    Test that Document's first-level fields take precedence over meta fields when flattening the dictionary
    representation.
    """

    doc = Document(content="from-content", score=0.9, meta={"content": "from-meta", "score": 0.5, "source": "web"})

    flat_dict = doc.to_dict(flatten=True)

    # First-level fields should take precedence
    assert flat_dict["content"] == "from-content"
    assert flat_dict["score"] == 0.9
    # Meta-only fields should be preserved
    assert flat_dict["source"] == "web"


def test_from_dict():
    assert Document.from_dict({}) == Document()


def test_from_dict_with_parameters():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "blob": {"data": list(blob_data), "mime_type": "text/markdown", "meta": {"text": "test text"}},
            "meta": {"text": "test text"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
        }
    ) == Document(
        content="test text",
        blob=ByteStream(blob_data, mime_type="text/markdown", meta={"text": "test text"}),
        meta={"text": "test text"},
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
    )


def test_from_dict_does_not_mutate_input():
    blob_data = b"some bytes"
    data = {
        "content": "test text",
        "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
        "score": 0.812,
        "embedding": [0.1, 0.2, 0.3],
        "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
        "date": "10-10-2023",
        "type": "article",
    }
    original_data = deepcopy(data)

    assert Document.from_dict(data) == Document(
        content="test text",
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
        meta={"date": "10-10-2023", "type": "article"},
    )
    assert data == original_data


def test_from_dict_does_not_mutate_input_with_explicit_meta():
    data = {"content": "test text", "meta": {"date": "10-10-2023", "type": "article"}, "score": 0.812}
    original_data = deepcopy(data)

    assert Document.from_dict(data) == Document(
        content="test text", meta={"date": "10-10-2023", "type": "article"}, score=0.812
    )
    assert data == original_data


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
        content="test text",
        content_type="text",
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],  # type: ignore
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
        id_hash_keys=["content"],
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        meta={"date": "10-10-2023", "type": "article"},
    )


def test_from_dict_with_flat_meta():
    blob_data = b"some bytes"
    assert Document.from_dict(
        {
            "content": "test text",
            "blob": {"data": list(blob_data), "mime_type": "text/markdown"},
            "score": 0.812,
            "embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]},
            "date": "10-10-2023",
            "type": "article",
        }
    ) == Document(
        content="test text",
        blob=ByteStream(blob_data, mime_type="text/markdown"),
        score=0.812,
        embedding=[0.1, 0.2, 0.3],
        sparse_embedding=SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3]),
        meta={"date": "10-10-2023", "type": "article"},
    )


def test_from_dict_with_flat_and_non_flat_meta():
    with pytest.raises(ValueError, match="Pass either the 'meta' parameter or flattened metadata keys"):
        Document.from_dict(
            {
                "content": "test text",
                "blob": {"data": list(b"some bytes"), "mime_type": "text/markdown"},
                "score": 0.812,
                "meta": {"test": 10},
                "embedding": [0.1, 0.2, 0.3],
                "date": "10-10-2023",
                "type": "article",
            }
        )


def test_from_dict_with_dataframe():
    """
    Test for legacy support of Document.from_dict() with dataframe field.

    Test that Document.from_dict() can properly deserialize a Document dictionary obtained with
    document.to_dict(flatten=False) in haystack-ai<=2.10.0.
    We make sure that Document.from_dict() does not raise an error and that dataframe is skipped (legacy field).
    """

    # Document dictionary obtained with document.to_dict(flatten=False) in haystack-ai<=2.10.0
    doc_dict = {
        "id": "my_id",
        "content": "my_content",
        "dataframe": None,
        "blob": None,
        "meta": {"key": "value"},
        "score": None,
        "embedding": None,
        "sparse_embedding": None,
    }

    doc = Document.from_dict(doc_dict)

    assert doc.id == "my_id"
    assert doc.content == "my_content"
    assert doc.meta == {"key": "value"}
    assert doc.score is None
    assert doc.embedding is None
    assert doc.sparse_embedding is None

    assert not hasattr(doc, "dataframe")


def test_content_type():
    assert Document(content="text").content_type == "text"

    with pytest.raises(ValueError):
        _ = Document().content_type


def test_no_warning_on_init():
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        Document(content="test")


def test_warn_on_inplace_mutation():
    doc = Document(content="test")
    with pytest.warns(Warning, match="dataclasses.replace"):
        doc.content = "other"

def test_document_empty_string_and_none_content_have_different_ids():
    """
    Regression test: Document(content="") and Document(content=None)
    must not produce the same ID. Previously, `self.content or None`
    coerced "" to None before hashing, causing a collision.
    """
    d_empty = Document(content="")
    d_none = Document(content=None)
    assert d_empty.id != d_none.id, (
        "Document with empty string content and Document with None content "
        "must have different IDs to avoid silent data loss in document stores."
    )


def test_document_empty_string_id_is_stable():
    """Document(content='') should always produce the same ID."""
    d1 = Document(content="")
    d2 = Document(content="")
    assert d1.id == d2.id


def test_document_none_content_id_is_stable():
    """Document(content=None) should always produce the same ID."""
    d1 = Document(content=None)
    d2 = Document(content=None)
    assert d1.id == d2.id


def test_document_store_accepts_empty_and_none_content_documents():
    """
    Both Document(content='') and Document(content=None) should be
    writable to the same store without DuplicateDocumentError.
    """
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    store = InMemoryDocumentStore()
    store.write_documents([
        Document(content=""),
        Document(content=None),
    ])
    assert store.count_documents() == 2