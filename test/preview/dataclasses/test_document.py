from pathlib import Path
import dataclasses
import textwrap

import pytest
import pandas as pd
import numpy as np

from haystack.preview import Document
from haystack.preview.dataclasses.document import _create_id, DocumentEncoder, DocumentDecoder


@pytest.mark.unit
def test_document_is_immutable():
    doc = Document(content="test content")
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.content = "won't work"


@pytest.mark.unit
def test_content_type_unknown():
    with pytest.raises(ValueError, match="Content type unknown: 'other'"):
        Document(content="test content", content_type="other")


@pytest.mark.unit
def test_content_type_must_match_text():
    Document(content="test content")
    Document(content="test content", content_type="text")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content="test content", content_type="table")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content="test content", content_type="image")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content="test content", content_type="audio")


@pytest.mark.unit
def test_content_type_must_match_table():
    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content=pd.DataFrame([1, 2]), content_type="text")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content=pd.DataFrame([1, 2]), content_type="image")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content=pd.DataFrame([1, 2]), content_type="audio")


@pytest.mark.unit
def test_content_type_must_match_image_audio():
    # Mimetypes are not checked - yet
    Document(content=Path(__file__), content_type="image")
    Document(content=Path(__file__), content_type="audio")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content=Path(__file__), content_type="text")

    with pytest.raises(ValueError, match="does not match the content type"):
        Document(content=Path(__file__), content_type="table")


@pytest.mark.unit
def test_id_hash_keys_require_metadata():
    with pytest.raises(ValueError, match="must be present in the metadata of the Document if you want to use it"):
        Document(content="test content", id_hash_keys=["something"])


@pytest.mark.unit
def test_init_document_same_meta_as_main_fields():
    """
    This is forbidden to prevent later issues with `Document.flatten()`
    """
    with pytest.raises(ValueError, match="score"):
        Document(content="test content", metadata={"score": "10/10"})


@pytest.mark.unit
def test_basic_equality_type_mismatch():
    doc = Document(content="test content")
    assert doc != "test content"


@pytest.mark.unit
def test_simple_table_document_equality():
    doc1 = Document(content=pd.DataFrame([1, 2]), content_type="table")
    doc2 = Document(content=pd.DataFrame([1, 2]), content_type="table")
    assert doc1 == doc2


@pytest.mark.unit
def test_simple_image_document_equality():
    doc1 = Document(content=Path(__file__).parent / "test_files" / "apple.jpg", content_type="image")
    doc2 = Document(content=Path(__file__).parent / "test_files" / "apple.jpg", content_type="image")
    assert doc1 == doc2


@pytest.mark.unit
def test_equality_with_embeddings():
    doc1 = Document(content="test content", embedding=np.array([10, 10]))
    doc2 = Document(content="test content", embedding=np.array([10, 10]))
    assert doc1 == doc2


@pytest.mark.unit
def test_equality_with_embeddings_shape_check():
    doc1 = Document(content="test content", embedding=np.array([10, 10]))
    doc2 = Document(content="test content", embedding=np.array([[[10, 10]]]))
    assert doc1 != doc2


@pytest.mark.unit
def test_equality_with_scores():
    doc1 = Document(content="test content", score=100)
    doc2 = Document(content="test content", score=100)
    assert doc1 == doc2


@pytest.mark.unit
def test_equality_with_simple_metadata():
    doc1 = Document(content="test content", metadata={"value": 1, "another": "value"})
    doc2 = Document(content="test content", metadata={"value": 1, "another": "value"})
    assert doc1 == doc2


@pytest.mark.unit
def test_equality_with_different_metadata():
    doc1 = Document(content="test content", metadata={"value": 1})
    doc2 = Document(content="test content", metadata={"value": 1, "another": "value"})
    assert doc1 != doc2


@pytest.mark.unit
def test_equality_with_nested_metadata():
    doc1 = Document(content="test content", metadata={"value": {"another": "value"}})
    doc2 = Document(content="test content", metadata={"value": {"another": "value"}})
    assert doc1 == doc2


@pytest.mark.unit
def test_equality_with_metadata_with_objects():
    class TestObject:
        def __eq__(self, other):
            if type(self) == type(other):
                return True

    doc1 = Document(
        content="test content", metadata={"value": np.array([0, 1, 2]), "path": Path(__file__), "obj": TestObject()}
    )
    doc2 = Document(
        content="test content", metadata={"value": np.array([0, 1, 2]), "path": Path(__file__), "obj": TestObject()}
    )
    assert doc1 == doc2


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
    assert Document.from_dict(
        {
            "id": _create_id(classname=Document.__name__, content=pd.DataFrame([1, 2])),
            "content": pd.DataFrame([1, 2]),
            "content_type": "table",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=pd.DataFrame([1, 2]), content_type="table")


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
    assert Document.from_dict(
        {
            "id": _create_id(classname=Document.__name__, content=Path(__file__).parent / "test_files" / "apple.jpg"),
            "content": Path(__file__).parent / "test_files" / "apple.jpg",
            "content_type": "image",
            "metadata": {},
            "id_hash_keys": [],
            "score": None,
            "embedding": None,
        }
    ) == Document(content=Path(__file__).parent / "test_files" / "apple.jpg", content_type="image")


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


@pytest.mark.unit
def test_default_text_document_to_json():
    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc_1 = Document(content="test content").to_json(indent=4).strip()
    doc_2 = textwrap.dedent(
        """    {
        "id": \""""
        + doc_id
        + """\",
        "content": "test content",
        "content_type": "text",
        "metadata": {},
        "id_hash_keys": [],
        "score": null,
        "embedding": null
    }"""
    ).strip()
    assert doc_1 == doc_2


@pytest.mark.unit
def test_default_text_document_from_json():
    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc_1 = Document(content="test content")
    doc_2 = Document.from_json(
        """{"id": \""""
        + doc_id
        + """\",
        "content": "test content",
        "content_type": "text",
        "metadata": {},
        "id_hash_keys": [],
        "score": null,
        "embedding": null
    }"""
    )
    assert doc_1 == doc_2


@pytest.mark.unit
def test_default_table_document_to_json():
    df = pd.DataFrame([1, 2])
    doc_id = _create_id(classname=Document.__name__, content=df)
    doc_1 = Document(content=df, content_type="table").to_json(indent=4).strip()
    doc_2 = textwrap.dedent(
        """    {
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
    }"""
    ).strip()
    assert doc_1 == doc_2


@pytest.mark.unit
def test_default_table_document_from_json():
    df = pd.DataFrame([1, 2])
    doc_id = _create_id(classname=Document.__name__, content=pd.DataFrame([1, 2]))
    doc_1 = Document(content=df, content_type="table")
    doc_2 = Document.from_json(
        """{
        "id": \""""
        + doc_id
        + """\",
        "content": \""""
        + pd.DataFrame([1, 2]).to_json().replace('"', '\\"')
        + """\",
        "content_type": "table",
        "metadata": {},
        "id_hash_keys": [],
        "score": null,
        "embedding": null
    }"""
    )
    assert doc_1 == doc_2


@pytest.mark.unit
def test_default_image_document_to_json():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    doc_id = _create_id(classname=Document.__name__, content=path)
    doc_1 = Document(content=path, content_type="image").to_json(indent=4).strip()
    doc_2 = textwrap.dedent(
        """    {
        "id": \""""
        + doc_id
        + """\",
        "content": \""""
        + str(path.absolute()).replace("\\", "\\\\")
        + """\",
        "content_type": "image",
        "metadata": {},
        "id_hash_keys": [],
        "score": null,
        "embedding": null
    }"""
    ).strip()
    assert doc_1 == doc_2


@pytest.mark.unit
def test_default_image_document_from_json():
    path = Path(__file__).parent / "test_files" / "apple.jpg"
    doc_id = _create_id(classname=Document.__name__, content=path)
    doc_1 = Document(content=path, content_type="image")
    doc_2 = Document.from_json(
        """{"id": \""""
        + doc_id
        + """\",
        "content": \""""
        + str(path.absolute()).replace("\\", "\\\\")
        + """\",
        "content_type": "image",
        "metadata": {},
        "id_hash_keys": [],
        "score": null,
        "embedding": null
    }"""
    )
    assert doc_1 == doc_2


@pytest.mark.unit
def test_full_document_to_json(tmp_path):
    class TestClass:
        def __repr__(self):
            return "<the object>"

    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc_1 = Document(
        content="test content",
        metadata={"some object": TestClass(), "a path": tmp_path / "test.txt"},
        embedding=np.array([1, 2, 3, 4]),
    )

    doc_json = doc_1.to_json(indent=4).strip()
    doc_2 = textwrap.dedent(
        """    {
        "id": \""""
        + doc_id
        + """\",
        "content": "test content",
        "content_type": "text",
        "metadata": {
            "some object": "<the object>",
            "a path": \""""
        + str((tmp_path / "test.txt").absolute()).replace("\\", "\\\\")
        + """\"
        },
        "id_hash_keys": [],
        "score": null,
        "embedding": [
            1,
            2,
            3,
            4
        ]
    }"""
    ).strip()
    assert doc_json == doc_2


@pytest.mark.unit
def test_full_document_from_json(tmp_path):
    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc_1 = Document(
        content="test content",
        metadata={"a path": str(tmp_path / "test.txt")},  # Paths are not cast back to Path objects
        embedding=np.array([1, 2, 3, 4]),
    )
    doc_2 = Document.from_json(
        """{"id": \""""
        + doc_id
        + """\",
        "content": "test content",
        "content_type": "text",
        "metadata": {
            "a path": \""""
        + str((tmp_path / "test.txt").absolute()).replace("\\", "\\\\")
        + """\"
        },
        "id_hash_keys": [],
        "score": null,
        "embedding": [
            1,
            2,
            3,
            4
        ]
    }"""
    )
    assert doc_1 == doc_2


@pytest.mark.unit
def test_to_json_custom_encoder(tmp_path):
    class SerializableTestClass:
        ...

    class TestEncoder(DocumentEncoder):
        def default(self, obj):
            if isinstance(obj, SerializableTestClass):
                return "<<CUSTOM ENCODING>>"
            return DocumentEncoder.default(self, obj)

    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc = Document(content="test content", metadata={"some object": SerializableTestClass()})
    doc_json = doc.to_json(indent=4, json_encoder=TestEncoder).strip()

    assert (
        doc_json
        == textwrap.dedent(
            """    {
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
    }"""
        ).strip()
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

    doc_id = _create_id(classname=Document.__name__, content="test content")
    doc = Document(content="test content", metadata={"some object": TestClass()})

    assert doc == Document.from_json(
        """    {
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
    }""",
        json_decoder=TestDecoder,
    )


@pytest.mark.unit
def test_flatten_text_document_no_meta():
    assert Document(content="test content").flatten() == {
        "id": _create_id(classname=Document.__name__, content="test content"),
        "content": "test content",
        "content_type": "text",
        "id_hash_keys": [],
        "score": None,
        "embedding": None,
    }


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
