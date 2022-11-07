import pytest

from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.schema import Document

from .test_base import DocumentStoreBaseTestAbstract


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


class TestWeaviateDocumentStore(DocumentStoreBaseTestAbstract):
    # Constants

    index_name = "DocumentsTest"

    @pytest.fixture
    def ds(self):
        return WeaviateDocumentStore(index=self.index_name, recreate_index=True)

    @pytest.fixture(scope="class")
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "year": "2020", "month": "01", "numbers": [2.0, 4.0]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "year": "2021", "month": "02", "numbers": [-2.0, -4.0]},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    id=get_uuid(),
                    content=f"A Baz Document {i}",
                    meta={"name": f"name_{i}", "month": "03"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_labels(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_id(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_filter(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_get_label_count(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_labels_duplicate(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_write_get_all_labels(self):
        pass

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        """
        Weaviate doesn't include documents if the field is missing,
        so we customize this test
        """
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        """
        Weaviate doesn't include documents if the field is missing,
        so we customize this test
        """
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$nin": ["2020", "2021", "n.a."]}})
        assert len(result) == 0

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, this doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0


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


@pytest.mark.weaviate
@pytest.mark.parametrize("similarity", ["cosine", "l2", "dot_product"])
def test_similarity_existing_index(tmp_path, similarity):
    """Testing non-matching similarity"""
    # create the document_store
    document_store = get_document_store("weaviate", tmp_path, similarity=similarity, recreate_index=True)

    # try to connect to the same document store but using the wrong similarity
    non_matching_similarity = "l2" if similarity == "cosine" else "cosine"
    with pytest.raises(ValueError, match=r"This index already exists in Weaviate with similarity .*"):
        document_store2 = get_document_store(
            "weaviate", tmp_path, similarity=non_matching_similarity, recreate_index=False
        )


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_cant_write_id_in_meta(document_store):
    with pytest.raises(ValueError, match='"meta" info contains duplicate key "id"'):
        document_store.write_documents([Document(content="test", meta={"id": "test-id"})])


@pytest.mark.weaviate
@pytest.mark.parametrize("document_store", ["weaviate"], indirect=True)
def test_cant_write_top_level_fields_in_meta(document_store):
    with pytest.raises(ValueError, match='"meta" info contains duplicate key "content"'):
        document_store.write_documents([Document(content="test", meta={"content": "test-id"})])
