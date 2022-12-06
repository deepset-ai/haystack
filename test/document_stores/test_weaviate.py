import pytest

from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.schema import Document

from .test_base import DocumentStoreBaseTestAbstract


import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from haystack.schema import Document

embedding_dim = 768


def get_uuid():
    return str(uuid.uuid4())


class TestWeaviateDocumentStore(DocumentStoreBaseTestAbstract):
    # Constants

    index_name = "DocumentsTest"

    @pytest.fixture
    def ds(self):
        return WeaviateDocumentStore(index=self.index_name, recreate_index=True, return_embedding=True)

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

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_labels_with_long_texts(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_no_answer(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_filter_aggregations(self):
        pass

    @pytest.mark.skip(reason="Weaviate does not support labels")
    @pytest.mark.integration
    def test_multilabel_meta_aggregations(self):
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

    @pytest.mark.integration
    def test_query_by_embedding(self, ds, documents):
        ds.write_documents(documents)

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32))
        assert len(docs) == 9

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32), top_k=1)
        assert len(docs) == 1

        docs = ds.query_by_embedding(np.random.rand(embedding_dim).astype(np.float32), filters={"name": ["name_1"]})
        assert len(docs) == 3

    @pytest.mark.integration
    def test_query(self, ds, documents):
        ds.write_documents(documents)

        query_text = "Foo"
        docs = ds.query(query_text)
        assert len(docs) == 3

        # BM25 retrieval WITH filters is not yet supported as of Weaviate v1.14.1
        # Should be from 1.18: https://github.com/semi-technologies/weaviate/issues/2393
        # docs = ds.query(query_text, filters={"name": ["name_1"]})
        # assert len(docs) == 1

        docs = ds.query(filters={"name": ["name_0"]})
        assert len(docs) == 3

        docs = ds.query(filters={"content": [query_text.lower()]})
        assert len(docs) == 3

        docs = ds.query(filters={"content": ["baz"]})
        assert len(docs) == 3

    @pytest.mark.integration
    def test_get_all_documents_unaffected_by_QUERY_MAXIMUM_RESULTS(self, ds, documents, monkeypatch):
        """
        Ensure `get_all_documents` works no matter the value of QUERY_MAXIMUM_RESULTS
        see https://github.com/deepset-ai/haystack/issues/2517
        """
        ds.write_documents(documents)
        monkeypatch.setattr(ds, "get_document_count", lambda **kwargs: 13_000)
        docs = ds.get_all_documents()
        assert len(docs) == 9

    @pytest.mark.integration
    def test_deleting_by_id_or_by_filters(self, ds, documents):
        ds.write_documents(documents)
        # This test verifies that deleting an object by its ID does not first require fetching all documents. This fixes
        # a bug, as described in https://github.com/deepset-ai/haystack/issues/2898
        ds.get_all_documents = MagicMock(wraps=ds.get_all_documents)

        assert ds.get_document_count() == 9

        # Delete a document by its ID. This should bypass the get_all_documents() call
        ds.delete_documents(ids=[documents[0].id])
        ds.get_all_documents.assert_not_called()
        assert ds.get_document_count() == 8

        ds.get_all_documents.reset_mock()
        # Delete a document with filters. Prove that using the filters will go through get_all_documents()
        ds.delete_documents(filters={"name": ["name_0"]})
        ds.get_all_documents.assert_called()
        assert ds.get_document_count() == 6

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["cosine", "l2", "dot_product"])
    def test_similarity_existing_index(self, similarity):
        """Testing non-matching similarity"""
        # create the document_store
        document_store = WeaviateDocumentStore(
            similarity=similarity, index=f"test_similarity_existing_index_{similarity}", recreate_index=True
        )

        # try to connect to the same document store but using the wrong similarity
        non_matching_similarity = "l2" if similarity == "cosine" else "cosine"
        with pytest.raises(ValueError, match=r"This index already exists in Weaviate with similarity .*"):
            document_store2 = WeaviateDocumentStore(
                similarity=non_matching_similarity,
                index=f"test_similarity_existing_index_{similarity}",
                recreate_index=False,
            )

    @pytest.mark.integration
    def test_cant_write_id_in_meta(self, ds):
        with pytest.raises(ValueError, match='"meta" info contains duplicate key "id"'):
            ds.write_documents([Document(content="test", meta={"id": "test-id"})])

    @pytest.mark.integration
    def test_cant_write_top_level_fields_in_meta(self, ds):
        with pytest.raises(ValueError, match='"meta" info contains duplicate key "content"'):
            ds.write_documents([Document(content="test", meta={"content": "test-id"})])
