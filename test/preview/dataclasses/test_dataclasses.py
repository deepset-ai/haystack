import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from mmh3 import hash128

from haystack.preview import Document


def test_default_text_document_to_dict():
    assert Document(content="test content").to_dict() == {
        "id": "{:02x}".format(hash128(":".join([Document.__name__, "test content"]), signed=False)),
        "content": "test content",
        "content_type": "text",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_default_text_document_from_dict():
    assert Document.from_dict(
        {
            "id": "{:02x}".format(hash128(":".join([Document.__name__, "test content"]), signed=False)),
            "content": "test content",
            "content_type": "text",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content="test content")


def test_default_table_document_to_dict():
    df = pd.DataFrame([1, 2])
    dictionary = Document(content=df, content_type="table").to_dict()

    dataframe = dictionary.pop("content")
    assert dataframe.equals(df)

    assert dictionary == {
        "id": "{:02x}".format(hash128(":".join([Document.__name__, str(df)]), signed=False)),
        "content_type": "table",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_default_table_document_from_dict():
    df = pd.DataFrame([1, 2])
    assert Document.from_dict(
        {
            "id": "{:02x}".format(hash128(":".join([Document.__name__, str(df)]), signed=False)),
            "content": df,
            "content_type": "table",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=df, content_type="table")


def test_default_image_document_to_dict():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    assert Document(content=path, content_type="image").to_dict() == {
        "id": "{:02x}".format(hash128(":".join([Document.__name__, str(path)]), signed=False)),
        "content": path,
        "content_type": "image",
        "metadata": {},
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


def test_default_image_document_from_dict():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    assert Document.from_dict(
        {
            "id": "{:02x}".format(hash128(":".join([Document.__name__, str(path)]), signed=False)),
            "content": path,
            "content_type": "image",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=path, content_type="image")


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
        "id": "{:02x}".format(hash128(":".join([Document.__name__, "test content", "10"]), signed=False)),
        "content": "test content",
        "content_type": "text",
        "metadata": {"some": "values", "test": 10},
        "id_hash_keys": ["test"],
        "score": 0.99,
    }


def test_document_with_most_attributes_from_dict():
    embedding = np.zeros([10, 10])
    assert Document.from_dict(
        {
            "id": "{:02x}".format(hash128(":".join([Document.__name__, "test content", "10"]), signed=False)),
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
