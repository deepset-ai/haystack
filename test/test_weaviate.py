import numpy as np
import pytest
from haystack.schema import Document
from conftest import get_document_store
import uuid

embedding_dim = 768


def get_uuid():
    return str(uuid.uuid4())


DOCUMENTS = [
    {"content": "text1", "id":"not a correct uuid", "key": "a"},
    {"content": "text2", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text3", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text4", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
    {"content": "text5", "id":get_uuid(), "key": "b", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
]

DOCUMENTS_XS = [
        # current "dict" format for a document
        {"content": "My name is Carla and I live in Berlin", "id":get_uuid(), "meta": {"metafield": "test1", "name": "filename1"}, "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        # meta_field at the top level for backward compatibility
        {"content": "My name is Paul and I live in New York", "id":get_uuid(), "metafield": "test2", "name": "filename2", "embedding": np.random.rand(embedding_dim).astype(np.float32)},
        # Document object for a doc
        Document(content="My name is Christelle and I live in Paris", id=get_uuid(), meta={"metafield": "test3", "name": "filename3"}, embedding=np.random.rand(embedding_dim).astype(np.float32))
    ]


@pytest.fixture(params=["weaviate"])
def document_store_with_docs(request):
    document_store = get_document_store(request.param)
    document_store.write_documents(DOCUMENTS_XS)
    yield document_store
    document_store.delete_documents()


@pytest.fixture(params=["weaviate"])
def document_store(request):
    document_store = get_document_store(request.param)
    yield document_store
    document_store.delete_documents()


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
@pytest.mark.parametrize("batch_size", [2])
def test_weaviate_write_docs(document_store, batch_size):
    # Write in small batches
    for i in range(0, len(DOCUMENTS), batch_size):
        document_store.write_documents(DOCUMENTS[i: i + batch_size])

    documents_indexed = document_store.get_all_documents()
    assert len(documents_indexed) == len(DOCUMENTS)

    documents_indexed = document_store.get_all_documents(batch_size=batch_size)
    assert len(documents_indexed) == len(DOCUMENTS)


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query_by_embedding(document_store_with_docs):
    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32))
    assert len(docs) == 3

    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32),
                                                       top_k=1)
    assert len(docs) == 1

    docs = document_store_with_docs.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32),
                                                       filters = {"name": ['filename2']})
    assert len(docs) == 1

@pytest.mark.weaviate
@pytest.mark.parametrize("document_store_with_docs", ["weaviate"], indirect=True)
def test_query(document_store_with_docs):
    query_text = 'My name is Carla and I live in Berlin'
    with pytest.raises(Exception):
        docs = document_store_with_docs.query(query_text)

    docs = document_store_with_docs.query(filters = {"name": ['filename2']})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"content":[query_text.lower()]})
    assert len(docs) == 1

    docs = document_store_with_docs.query(filters={"content":['live']})
    assert len(docs) == 3
