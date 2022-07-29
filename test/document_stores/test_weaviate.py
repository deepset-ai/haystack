import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from haystack.schema import Document
from ..conftest import get_document_store

embedding_dim = 768


def get_uuid():
    return str(uuid.uuid4())


DOCUMENTS = [
    {"content": "text1", "id": "not a correct uuid", "key": "a"},
    {"content": "text2", "id": get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text3", "id": get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text4", "id": get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text5", "id": get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
]

DOCUMENTS_XS = [
    # current "dict" format for a document
    {
        "content": "My name is Carla and I live in Berlin",
        "id": get_uuid(),
        "meta": {"metafield": "test1", "name": "filename1"},
        "embedding": np.random.rand(embedding_dim).astype(np.float32),
    },
    # meta_field at the top level for backward compatibility
    {
        "content": "My name is Paul and I live in New York",
        "id": get_uuid(),
        "metafield": "test2",
        "name": "filename2",
        "embedding": np.random.rand(embedding_dim).astype(np.float32),
    },
    # Document object for a doc
    Document(
        content="My name is Christelle and I live in Paris",
        id=get_uuid(),
        meta={"metafield": "test3", "name": "filename3"},
        embedding=np.random.rand(embedding_dim).astype(np.float32),
    ),
]


@pytest.fixture(params=["weaviate"])
def document_store_with_docs(request, tmp_path):
    document_store = get_document_store(request.param, tmp_path=tmp_path)
    document_store.write_documents(DOCUMENTS_XS)
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.fixture(params=["weaviate"])
def document_store(request, tmp_path):
    document_store = get_document_store(request.param, tmp_path=tmp_path)
    yield document_store
    document_store.delete_index(document_store.index)


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
@pytest.mark.parametrize("batch_size", [2])
def test_weaviate_write_docs(document_store, batch_size):
    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i : i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    documents_indexed = document_store.get_all_documents(batch_size=batch_size)
    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query_by_embedding(document_store_with_docs):
    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32))
    assert len(docs) == 3

    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32), top_k=1)
    assert len(docs) == 1

    docs = document_store_with_docs.query_by_embedding(
        np.random.rand(embedding_dim).astype(np.float32), filters={"name": ["filename2"]}
    )
    assert len(docs) == 1


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query(document_store_with_docs):
    query_text = "My name is Carla and I live in Berlin"
    docs = document_store_with_docs.query(query_text)
    assert len(docs) == 3

    # BM25 retrieval WITH filters is not yet supported as of Weaviate v1.14.1
    with pytest.raises(Exception):
        docs = document_store_with_docs.query(query_text, filters={"name": ["filename2"]})

    docs = document_store_with_docs.query(filters={"name": ["filename2"]})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"content": [query_text.lower()]})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"content": ["live"]})
    assert len(docs) == 3


@pytest.mark.weaviate
def test_get_all_documents_unaffected_by_QUERY_MAXIMUM_RESULTS(document_store_with_docs, monkeypatch):
    """
    Ensure `get_all_documents` works no matter the value of QUERY_MAXIMUM_RESULTS
    see https://github.com/deepset-ai/haystack/issues/2517
    """
    monkeypatch.setattr(document_store_with_docs, "get_document_count", lambda **kwargs: 13_000)
    docs = document_store_with_docs.get_all_documents()
    assert len(docs) == 3


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_deleting_by_id_or_by_filters(document_store_with_docs):
    # This test verifies that deleting an object by its ID does not first require fetching all documents. This fixes
    # a bug, as described in https://github.com/deepset-ai/haystack/issues/2898
    document_store_with_docs.get_all_documents = MagicMock(wraps=document_store_with_docs.get_all_documents)

    assert document_store_with_docs.get_document_count() == 3

    # Delete a document by its ID. This should bypass the get_all_documents() call
    document_store_with_docs.delete_documents(ids=[DOCUMENTS_XS[0]["id"]])
    document_store_with_docs.get_all_documents.assert_not_called()
    assert document_store_with_docs.get_document_count() == 2

    document_store_with_docs.get_all_documents.reset_mock()
    # Delete a document with filters. Prove that using the filters will go through get_all_documents()
    document_store_with_docs.delete_documents(filters={"name": ["filename2"]})
    document_store_with_docs.get_all_documents.assert_called()
    assert document_store_with_docs.get_document_count() == 1
