from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import pytest

from haystack.preview import Document
from haystack.preview.dataclasses.document import _create_id


@pytest.mark.unit
def test_init_document_same_meta_as_main_fields():
    """
    This is forbidden to prevent later issues with `Document.flatten()`
    """
    with pytest.raises(ValueError, match="score"):
        Document(content="test content", metadata={"score": "10/10"})


@pytest.mark.unit
def test_default_text_document_to_dict():
    assert Document(content="test content").to_dict() == {
        "id": _create_id(classname=Document.__name__, content="test content"),
        "content": "test content",
        "content_type": "text",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_default_text_document_from_dict():
    assert Document.from_dict(
        {
            "id": _create_id(classname=Document.__name__, content="test content"),
            "content": "test content",
            "content_type": "text",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content="test content")


@pytest.mark.unit
def test_default_table_document_to_dict():
    df = pd.DataFrame([1, 2])
    dictionary = Document(content=df, content_type="table").to_dict()

    dataframe = dictionary.pop("content")
    assert dataframe.equals(df)

    assert dictionary == {
        "id": _create_id(classname=Document.__name__, content=df),
        "content_type": "table",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_default_table_document_from_dict():
    df = pd.DataFrame([1, 2])
    assert Document.from_dict(
        {
            "id": _create_id(classname=Document.__name__, content=df),
            "content": df,
            "content_type": "table",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=df, content_type="table")


@pytest.mark.unit
def test_default_image_document_to_dict():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    assert Document(content=path, content_type="image").to_dict() == {
        "id": _create_id(classname=Document.__name__, content=path),
        "content": path,
        "content_type": "image",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
def test_default_image_document_from_dict():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    assert Document.from_dict(
        {
            "id": _create_id(classname=Document.__name__, content=path),
            "content": path,
            "content_type": "image",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=path, content_type="image")


@pytest.mark.unit
def test_document_with_most_attributes_to_dict():
    """
    This tests also id_hash_keys
    """
    doc = Document(
        content="test content",
        content_type="text",
        metadata={"some": "values", "test": 10},
        id_hash_keys=["test"],
        score=0.99,
        embedding=np.zeros([10, 10]),
    )
    dictionary = doc.to_dict()

    embedding = dictionary.pop("embedding")
    assert (embedding == np.zeros([10, 10])).all()

    assert dictionary == {
        "id": _create_id(
            classname=Document.__name__,
            content="test content",
            id_hash_keys=["test"],
            metadata={"some": "values", "test": 10},
        ),
        "content": "test content",
        "content_type": "text",
        "metadata": {"some": "values", "test": 10},
        "id_hash_keys": ["test"],
        "score": 0.99,
    }


@pytest.mark.unit
def test_document_with_most_attributes_from_dict():
    embedding = np.zeros([10, 10])
    assert Document.from_dict(
        {
            "id": _create_id(
                classname=Document.__name__,
                content="test content",
                id_hash_keys=["test"],
                metadata={"some": "values", "test": 10},
            ),
            "content": "test content",
            "content_type": "text",
            "metadata": {"some": "values", "test": 10},
            "id_hash_keys": ["test"],
            "score": 0.99,
            "embedding": embedding,
        }
    ) == Document(
        content="test content",
        content_type="text",
        metadata={"some": "values", "test": 10},
        id_hash_keys=["test"],
        score=0.99,
        embedding=embedding,
    )


def test_flatten_text_document_no_meta():
    assert Document(content="test content").flatten() == {
        "id": _create_id(classname=Document.__name__, content="test content"),
        "content": "test content",
        "content_type": "text",
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_flatten_text_document():
    assert Document(content="test content", metadata={"name": "document name", "page": 123}).flatten() == {
        "id": _create_id(classname=Document.__name__, content="test content"),
        "content": "test content",
        "content_type": "text",
        "name": "document name",
        "page": 123,
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_flatten_table_document():
    df = pd.DataFrame([1, 2])
    flat = Document(content=df, content_type="table", metadata={"table-name": "table title", "section": 3}).flatten()

    dataframe = flat.pop("content")
    assert dataframe.equals(df)
    assert flat == {
        "id": _create_id(classname=Document.__name__, content=df),
        "content_type": "table",
        "table-name": "table title",
        "section": 3,
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_flatten_image_document():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    assert Document(
        content=path, content_type="image", metadata={"image title": "The Apple", "year": 1993}
    ).flatten() == {
        "id": _create_id(classname=Document.__name__, content=path),
        "content": path,
        "content_type": "image",
        "image title": "The Apple",
        "year": 1993,
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }
