from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from haystack.preview import Document
from haystack.preview.dataclasses.document import _create_id, DocumentEncoder, DocumentDecoder


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


def test_default_text_document_to_json():
    doc_id = _create_id(classname=Document.__name__, content="test content")
    assert (
        Document(content="test content").to_json(indent=4).strip()
        == """{
    "id": \""""
        + doc_id
        + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
""".strip()
    )


def test_default_text_document_from_json():
    doc_id = _create_id(classname=Document.__name__, content="test content")
    assert Document(content="test content") == Document.from_json(
        """{
    "id": \""""
        + doc_id
        + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
"""
    )


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


def test_default_table_document_to_json():
    df = pd.DataFrame([1, 2])
    doc_id = _create_id(classname=Document.__name__, content=df)
    assert (
        Document(content=df, content_type="table").to_json(indent=4).strip()
        == """{
    "id": \""""
        + doc_id
        + """\",
    "content": \""""
        + df.to_json().replace('"', '\\"')
        + """\",
    "content_type": "table",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
""".strip()
    )


# Waiting for https://github.com/deepset-ai/haystack/pull/4860
def test_default_table_document_from_json():
    df = pd.DataFrame([1, 2])
    doc_id = _create_id(classname=Document.__name__, content=df)

    ref_doc = Document(content=df, content_type="table")
    loaded_doc = Document.from_json(
        """{
    "id": \""""
        + doc_id
        + """\",
    "content": \""""
        + df.to_json().replace('"', '\\"')
        + """\",
    "content_type": "table",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
"""
    )
    assert df.equals(loaded_doc.content)
    assert loaded_doc == ref_doc


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


def test_default_image_document_to_json():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    doc_id = _create_id(classname=Document.__name__, content=path)
    assert (
        Document(content=path, content_type="image").to_json(indent=4).strip()
        == """{
    "id": \""""
        + doc_id
        + """\",
    "content": \""""
        + str(path)
        + """\",
    "content_type": "image",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
""".strip()
    )


def test_default_image_document_from_json():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    doc_id = _create_id(classname=Document.__name__, content=path)

    ref_doc = Document(content=path, content_type="image")
    loaded_doc = Document.from_json(
        """{
    "id": \""""
        + doc_id
        + """\",
    "content": \""""
        + str(path)
        + """\",
    "content_type": "image",
    "metadata": {},
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
"""
    )
    assert loaded_doc == ref_doc


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


def test_document_with_most_attributes_to_json():
    """
    This tests also id_hash_keys
    """
    doc = Document(
        content="test content",
        content_type="text",
        metadata={"some": "values", "test": 10},
        id_hash_keys=["test"],
        score=0.99,
        embedding=np.array([10, 10]),
    )
    dictionary = doc.to_json(indent=4)

    doc_id = _create_id(
        classname=Document.__name__,
        content="test content",
        id_hash_keys=["test"],
        metadata={"some": "values", "test": 10},
    )
    assert (
        dictionary
        == """{
    "id": \""""
        + doc_id
        + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {
        "some": "values",
        "test": 10
    },
    "id_hash_keys": [
        "test"
    ],
    "score": 0.99,
    "embedding": [
        10,
        10
    ]
}"""
    )


# Waiting for https://github.com/deepset-ai/haystack/pull/4860
def test_document_with_most_attributes_from_json():
    doc_id = _create_id(
        classname=Document.__name__,
        content="test content",
        id_hash_keys=["test"],
        metadata={"some": "values", "test": 10},
    )
    assert (
        Document.from_json(
            """{
    "id": \""""
            + doc_id
            + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {
        "some": "values",
        "test": 10
    },
    "id_hash_keys": [
        "test"
    ],
    "score": 0.99,
    "embedding": [
        10,
        10
    ]
}"""
        )
        == Document(
            content="test content",
            content_type="text",
            metadata={"some": "values", "test": 10},
            id_hash_keys=["test"],
            score=0.99,
            embedding=np.array([10, 10]),
        )
    )


def test_to_json_custom_encoder():
    class TestClass:
        ...

    class TestEncoder(DocumentEncoder):
        def default(self, obj):
            if isinstance(obj, TestClass):
                return "<<CUSTOM ENCODING>>"
            return DocumentEncoder.default(self, obj)

    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc = Document(content="test content", metadata={"some object": TestClass()})

    assert (
        doc.to_json(indent=4, json_encoder=TestEncoder).strip()
        == """{
    "id": \""""
        + doc_id
        + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {
        "some object": "<<CUSTOM ENCODING>>"
    },
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
""".strip()
    )


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

    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc = Document(content="test content", metadata={"some object": TestClass()})

    assert doc == Document.from_json(
        """{
    "id": \""""
        + doc_id
        + """\",
    "content": "test content",
    "content_type": "text",
    "metadata": {
        "some object": "<<CUSTOM ENCODING>>"
    },
    "id_hash_keys": [],
    "score": null,
    "embedding": null
}
""",
        json_decoder=TestDecoder,
    )
