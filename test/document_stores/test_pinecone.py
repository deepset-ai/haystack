from typing import List, Union, Dict, Any

import os
import numpy as np
from inspect import getmembers, isclass, isfunction
from unittest.mock import MagicMock

import pytest

from haystack.document_stores.pinecone import pinecone
from haystack.document_stores.pinecone import PineconeDocumentStore
from haystack.schema import Document
from haystack.errors import FilterError, PineconeDocumentStoreError
from haystack.testing import DocumentStoreBaseTestAbstract

from ..mocks import pinecone as pinecone_mock
from ..conftest import MockBaseRetriever

# Set metadata fields used during testing for PineconeDocumentStore meta_config
META_FIELDS = ["meta_field", "name", "date", "numeric_field", "odd_document"]


class TestPineconeDocumentStore(DocumentStoreBaseTestAbstract):
    # Fixtures

    @pytest.fixture
    def ds(self, monkeypatch, request) -> PineconeDocumentStore:
        """
        This fixture provides an empty document store and takes care of cleaning up after each test
        """
        # If it's a unit test, mock Pinecone
        if request.config.getoption("--mock-pinecone"):
            for fname, function in getmembers(pinecone_mock, isfunction):
                monkeypatch.setattr(f"pinecone.{fname}", function, raising=False)
            for cname, class_ in getmembers(pinecone_mock, isclass):
                monkeypatch.setattr(f"pinecone.{cname}", class_, raising=False)

        return PineconeDocumentStore(
            api_key=os.environ.get("PINECONE_API_KEY") or "fake-pinecone-test-key",
            embedding_dim=768,
            embedding_field="embedding",
            index="haystack_tests",
            similarity="cosine",
            recreate_index=True,
            metadata_config={"indexed": META_FIELDS},
        )

    @pytest.fixture
    def doc_store_with_docs(self, ds: PineconeDocumentStore, documents: List[Document]) -> PineconeDocumentStore:
        """
        This fixture provides a pre-populated document store and takes care of cleaning up after each test
        """
        ds.write_documents(documents)
        return ds

    @pytest.fixture
    def mocked_ds(self):
        class DSMock(PineconeDocumentStore):
            pass

        pinecone.init = MagicMock()
        DSMock._create_index = MagicMock()
        mocked_ds = DSMock(api_key="MOCK")

        return mocked_ds

    @pytest.fixture
    def docs_all_formats(self) -> List[Union[Document, Dict[str, Any]]]:
        return [
            # metafield at the top level for backward compatibility
            {
                "content": "My name is Paul and I live in New York",
                "meta_field": "test-1",
                "name": "file_1.txt",
                "date": "2019-10-01",
                "numeric_field": 5.0,
                "odd_document": True,
                "year": "2021",
                "month": "02",
            },
            # "dict" format
            {
                "content": "My name is Carla and I live in Berlin",
                "meta": {
                    "meta_field": "test-2",
                    "name": "file_2.txt",
                    "date": "2020-03-01",
                    "numeric_field": 5.5,
                    "odd_document": False,
                    "year": "2021",
                    "month": "02",
                },
            },
            # Document object
            Document(
                content="My name is Christelle and I live in Paris",
                meta={
                    "meta_field": "test-3",
                    "name": "file_3.txt",
                    "date": "2018-10-01",
                    "numeric_field": 4.5,
                    "odd_document": True,
                    "year": "2020",
                    "month": "02",
                },
            ),
            Document(
                content="My name is Camila and I live in Madrid",
                meta={
                    "meta_field": "test-4",
                    "name": "file_4.txt",
                    "date": "2021-02-01",
                    "numeric_field": 3.0,
                    "odd_document": False,
                    "year": "2020",
                },
            ),
            Document(
                content="My name is Matteo and I live in Rome",
                meta={
                    "meta_field": "test-5",
                    "name": "file_5.txt",
                    "date": "2019-01-01",
                    "numeric_field": 0.0,
                    "odd_document": True,
                    "year": "2020",
                },
            ),
            Document(
                content="My name is Adele and I live in London",
                meta={
                    "meta_field": "test-5",
                    "name": "file_5.txt",
                    "date": "2019-01-01",
                    "numeric_field": 0.0,
                    "odd_document": True,
                    "year": "2021",
                },
            ),
            # Without meta
            Document(content="My name is Ahmed and I live in Cairo"),
            Document(content="My name is Bruce and I live in Gotham"),
            Document(content="My name is Peter and I live in Quahog"),
        ]

    @pytest.fixture
    def documents(self, docs_all_formats: List[Union[Document, Dict[str, Any]]]) -> List[Document]:
        return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]

    #
    #  Tests
    #
    @pytest.mark.integration
    def test_doc_store_wrong_init(self):
        """
        This is just a failure check case.
        """
        try:
            _ = PineconeDocumentStore(
                api_key=os.environ.get("PINECONE_API_KEY") or "fake-pinecone-test-key",
                embedding_dim=768,
                pinecone_index="p_index",
                embedding_field="embedding",
                index="haystack_tests",
                similarity="cosine",
                metadata_config={"indexed": META_FIELDS},
            )
            assert False
        except PineconeDocumentStoreError as pe:
            assert "`pinecone_index` needs to be a `pinecone.Index` object" in pe.message

    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"year": {"$ne": "2020"}})
        assert len(result) == 3

    @pytest.mark.integration
    def test_get_label_count(self, ds, labels):
        with pytest.raises(NotImplementedError):
            ds.get_label_count()

    # NOTE: the PineconeDocumentStore behaves differently to the others when filters are applied.
    # While this should be considered a bug, the relative tests are skipped in the meantime

    @pytest.mark.skip
    @pytest.mark.integration
    def test_compound_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nin_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_documents_by_id_with_filters(self, ds, documents):
        pass

    # NOTE: labels metadata are not supported

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter(self, ds, labels):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_delete_labels_by_filter_id(self, ds, labels):
        pass

    @pytest.mark.skip
    @pytest.mark.integration
    def test_simplified_filters(self, ds, documents):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_labels_with_long_texts(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_no_answer(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_filter_aggregations(self):
        pass

    @pytest.mark.skip(reason="labels metadata are not supported")
    @pytest.mark.integration
    def test_multilabel_meta_aggregations(self):
        pass

    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_eq(self, doc_store_with_docs: PineconeDocumentStore):
        eq_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$eq": "test-1"}})
        normal_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": "test-1"})
        assert eq_docs == normal_docs

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_in(self, doc_store_with_docs: PineconeDocumentStore):
        in_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$in": ["test-1", "test-2", "n.a."]}})
        normal_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": ["test-1", "test-2", "n.a."]})
        assert in_docs == normal_docs

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_ne(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$ne": "test-1"}})
        assert all("test-1" != d.meta.get("meta_field", None) for d in retrieved_docs)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_nin(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(
            filters={"meta_field": {"$nin": ["test-1", "test-2", "n.a."]}}
        )
        assert {"test-1", "test-2"}.isdisjoint({d.meta.get("meta_field", None) for d in retrieved_docs})

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_gt(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$gt": 3.0}})
        assert all(d.meta["numeric_field"] > 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_gte(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$gte": 3.0}})
        assert all(d.meta["numeric_field"] >= 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_lt(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$lt": 3.0}})
        assert all(d.meta["numeric_field"] < 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_lte(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$lte": 3.0}})
        assert all(d.meta["numeric_field"] <= 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates(self, doc_store_with_docs: PineconeDocumentStore):
        filters = {"date": {"$lte": "2020-12-31", "$gte": "2019-01-01"}}

        with pytest.raises(FilterError, match=r"Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_other_field_explicit(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters = {
            "$and": {
                "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
                "name": {"$in": ["file_5.txt", "file_3.txt"]},
            }
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_other_field_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "name": ["file_5.txt", "file_3.txt"],
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_or_explicit(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters = {
            "$and": {
                "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
                "$or": {"name": {"$in": ["file_5.txt", "file_3.txt"]}, "numeric_field": {"$lte": 5.0}},
            }
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_or_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "$or": {"name": ["file_5.txt", "file_3.txt"], "numeric_field": {"$lte": 5.0}},
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_or_and_not_explicit(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters = {
            "$and": {
                "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
                "$or": {
                    "name": {"$in": ["file_5.txt", "file_3.txt"]},
                    "$and": {"numeric_field": {"$lte": 5.0}, "$not": {"meta_field": {"$eq": "test-2"}}},
                },
            }
        }
        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_dates_and_or_and_not_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "$or": {
                "name": ["file_5.txt", "file_3.txt"],
                "$and": {"numeric_field": {"$lte": 5.0}, "$not": {"meta_field": "test-2"}},
            },
        }
        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_nested_not(self, doc_store_with_docs: PineconeDocumentStore):
        # Test nested logical operations within "$not", important as we apply De Morgan's laws in Weaviatedocstore
        filters = {
            "$not": {
                "$or": {
                    "$and": {"numeric_field": {"$gt": 3.0}, "meta_field": {"$ne": "test-3"}},
                    "$not": {"date": {"$lt": "2020-01-01"}},
                }
            }
        }
        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]t' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_get_all_documents_extended_filter_compound_same_level_not(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        # Test same logical operator twice on same level, important as we apply De Morgan's laws in Weaviatedocstore
        filters = {
            "$or": [
                {"$and": {"meta_field": {"$in": ["test-1", "test-2"]}, "date": {"$gte": "2020-01-01"}}},
                {"$and": {"meta_field": {"$in": ["test-3", "test-4"]}, "date": {"$lt": "2020-01-01"}}},
            ]
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_multilayer_dict(self, doc_store_with_docs: PineconeDocumentStore):
        # Test that multilayer dict can be upserted
        multilayer_meta = {
            "parent1": {"parent2": {"parent3": {"child1": 1, "child2": 2}}},
            "meta_field": "multilayer-test",
        }
        doc = Document(
            content=f"Multilayered dict", meta=multilayer_meta, embedding=np.random.rand(768).astype(np.float32)
        )

        doc_store_with_docs.write_documents([doc])
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$eq": "multilayer-test"}})

        assert len(retrieved_docs) == 1
        assert retrieved_docs[0].meta == multilayer_meta

    @pytest.mark.unit
    def test_skip_validating_empty_embeddings(self, ds: PineconeDocumentStore):
        document = Document(id="0", content="test")
        retriever = MockBaseRetriever(document_store=ds, mock_document=document)
        ds.write_documents(documents=[document])
        ds._validate_embeddings_shape = MagicMock()

        ds.update_embeddings(retriever)
        ds._validate_embeddings_shape.assert_called_once()
        ds.update_embeddings(retriever, update_existing_embeddings=False)
        ds._validate_embeddings_shape.assert_called_once()

    @pytest.mark.integration
    def test_get_embedding_count(self, doc_store_with_docs: PineconeDocumentStore):
        """
        We expect 1 doc with an embeddings because all documents in already written in doc_store_with_docs contain no
        embeddings.
        """
        doc = Document(content=f"Doc with embedding", embedding=np.random.rand(768).astype(np.float32))
        doc_store_with_docs.write_documents([doc])
        assert doc_store_with_docs.get_embedding_count() == 1

    @pytest.mark.integration
    def test_get_document_count_after_write_doc_with_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        with an embedding is written to the document store.
        """
        # there are 9 docs in doc_store_with_docs (all without embeddings)
        initial_document_count = 9

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and also initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # document with embedding is written to doc_store_with_docs
        doc = Document(content=f"Doc with embedding", embedding=np.random.rand(768).astype(np.float32))
        doc_store_with_docs.write_documents([doc])

        # so we expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # but we expect initial_document_count documents without embeddings to be unchanged
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count

    @pytest.mark.integration
    def test_get_document_count_after_write_doc_without_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        without an embedding is written to the document store.
        """
        # there are 9 docs in doc_store_with_docs (all without embeddings)
        initial_document_count = 9

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and we also expect initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # document without embedding is written to doc_store_with_docs
        doc = Document(content=f"Doc without embedding")
        doc_store_with_docs.write_documents([doc])

        # we now expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # And we also expect initial_document_count + 1 documents without embeddings, because the document we just
        # wrote has no embeddings
        assert (
            doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count + 1
        )

    @pytest.mark.integration
    def test_get_document_count_after_delete_doc_with_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        with an embedding is deleted from the document store.
        """
        # there are 9 docs in doc_store_with_docs (all without embeddings)
        initial_document_count = 9

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and also initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # two documents with embedding are written to doc_store_with_docs
        doc_1 = Document(content=f"Doc with embedding 1", embedding=np.random.rand(768).astype(np.float32))
        doc_2 = Document(content=f"Doc with embedding 2", embedding=np.random.rand(768).astype(np.float32))
        doc_store_with_docs.write_documents([doc_1, doc_2])

        # total number is initial_document_count + 2
        assert doc_store_with_docs.get_document_count() == initial_document_count + 2

        # remove one of the documents with embedding
        all_embedding_docs = doc_store_with_docs.get_all_documents(namespace="vectors")
        doc_store_with_docs.delete_documents(ids=[all_embedding_docs[0].id])

        # since we deleted one doc, we expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # and we expect initial_document_count documents without embeddings
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count

    @pytest.mark.integration
    def test_get_document_count_after_delete_doc_without_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        without embedding is deleted from the document store.
        """
        # there are 9 docs in doc_store_with_docs (all without embeddings)
        initial_document_count = 9

        # therefore we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and also initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # two documents without embedding are written to doc_store_with_docs
        doc_1 = Document(content=f"Doc with embedding 1", embedding=None)
        doc_2 = Document(content=f"Doc with embedding 2", embedding=None)
        doc_store_with_docs.write_documents([doc_1, doc_2])

        # total number is initial_document_count + 2
        assert doc_store_with_docs.get_document_count() == initial_document_count + 2

        # remove one of the documents without embedding
        all_non_embedding_docs = doc_store_with_docs.get_all_documents(namespace="no-vectors")
        doc_store_with_docs.delete_documents(ids=[all_non_embedding_docs[0].id])

        # since we deleted one doc, we expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # and we expect initial_document_count +1 documents without embeddings as well
        assert (
            doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count + 1
        )

    @pytest.mark.unit
    def test_get_all_labels_legacy_document_id(self, ds, monkeypatch):
        monkeypatch.setattr(
            ds,
            "get_all_documents",
            MagicMock(
                return_value=[
                    Document.from_dict(
                        {
                            "content": "My name is Carla and I live in Berlin",
                            "content_type": "text",
                            "score": None,
                            "meta": {
                                "label-id": "d9256445-7b8a-4a33-a558-402ec84d6881",
                                "query": "query_1",
                                "label-is-correct-answer": False,
                                "label-is-correct-document": True,
                                "label-document-content": "My name is Carla and I live in Berlin",
                                "label-document-id": "a0747b83aea0b60c4b114b15476dd32d",
                                "label-no-answer": False,
                                "label-origin": "user-feedback",
                                "label-created-at": "2023-02-07 14:46:54",
                                "label-updated-at": None,
                                "label-pipeline-id": None,
                                "label-document-meta-meta_field": "test-2",
                                "label-document-meta-name": "file_2.txt",
                                "label-document-meta-date": "2020-03-01",
                                "label-document-meta-numeric_field": 5.5,
                                "label-document-meta-odd_document": False,
                                "label-document-meta-year": "2021",
                                "label-document-meta-month": "02",
                                "label-meta-name": "label_1",
                                "label-meta-year": "2021",
                                "label-answer-answer": "the answer is 1",
                                "label-answer-type": "extractive",
                                "label-answer-score": None,
                                "label-answer-context": None,
                                # legacy document_id answer
                                "label-answer-document-id": "a0747b83aea0b60c4b114b15476dd32d",
                                "label-answer-offsets-in-document-start": None,
                                "label-answer-offsets-in-document-end": None,
                                "label-answer-offsets-in-context-start": None,
                                "label-answer-offsets-in-context-end": None,
                            },
                            "id_hash_keys": ["content"],
                            "embedding": None,
                            "id": "d9256445-7b8a-4a33-a558-402ec84d6881",
                        }
                    )
                ]
            ),
        )

        labels = ds.get_all_labels()

        assert labels[0].answer.document_ids == ["a0747b83aea0b60c4b114b15476dd32d"]

    @pytest.mark.unit
    def test_split_overlap_meta(self, mocked_ds):
        """
        Tests that we can upload Docs with a _split_overlap_meta field to Pinecone as a JSON string
        and that the field is parsed correctly as dictionary when retrieved.
        """
        doc = Document(content="test", meta={"_split_overlap": [{"doc_id": "test_id", "range": (0, 10)}]}, id="test_id")
        # Test writing as JSON string
        mocked_ds.write_documents([doc])
        call_args = mocked_ds.pinecone_indexes["document"].upsert.call_args.kwargs
        assert list(call_args["vectors"])[0][2] == {
            "content": "test",
            "content_type": "text",
            "_split_overlap": '[{"doc_id": "test_id", "range": [0, 10]}]',
        }
        # Test retrieving as dict
        mocked_ds._get_all_document_ids = MagicMock(return_value=["test_id"])
        mocked_ds.pinecone_indexes["document"].fetch.return_value = {
            "vectors": {
                "test_id": {
                    "metadata": {
                        "_split_overlap": '[{"doc_id": "test_id", "range": [0, 10]}]',
                        "content": "test",
                        "content_type": "text",
                    }
                }
            }
        }
        retrieved_docs = mocked_ds.get_all_documents()
        assert retrieved_docs[0].meta["_split_overlap"] == [{"doc_id": "test_id", "range": [0, 10]}]
