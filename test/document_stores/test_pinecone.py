from typing import List, Union, Dict, Any

import os
from datetime import datetime
from inspect import getmembers, isclass, isfunction

import pytest

from haystack.document_stores.pinecone import PineconeDocumentStore
from haystack.schema import Document
from haystack.errors import FilterError


from ..mocks import pinecone as pinecone_mock
from ..conftest import SAMPLES_PATH


# Set metadata fields used during testing for PineconeDocumentStore meta_config
META_FIELDS = ["meta_field", "name", "date", "numeric_field", "odd_document"]


#
# FIXME This class should extend the base Document Store test class once it exists.
# At that point some of the fixtures will be duplicate, so review them.
#
class TestPineconeDocumentStore:

    # Fixtures

    @pytest.fixture
    def doc_store(self, monkeypatch, request) -> PineconeDocumentStore:
        """
        This fixture provides an empty document store and takes care of cleaning up after each test
        """
        # If it's a unit test, mock Pinecone
        if not "integration" in request.keywords:
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
    def doc_store_with_docs(self, doc_store: PineconeDocumentStore, docs: List[Document]) -> PineconeDocumentStore:
        """
        This fixture provides a pre-populated document store and takes care of cleaning up after each test
        """
        doc_store.write_documents(docs)
        return doc_store

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
                },
            ),
            # Without meta
            Document(content="My name is Ahmed and I live in Cairo"),
        ]

    @pytest.fixture
    def docs(self, docs_all_formats: List[Union[Document, Dict[str, Any]]]) -> List[Document]:
        return [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in docs_all_formats]

    #
    #  Tests
    #

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_eq(self, doc_store_with_docs: PineconeDocumentStore):
        eq_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$eq": "test-1"}})
        normal_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": "test-1"})
        assert eq_docs == normal_docs

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_in(self, doc_store_with_docs: PineconeDocumentStore):
        in_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$in": ["test-1", "test-2", "n.a."]}})
        normal_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": ["test-1", "test-2", "n.a."]})
        assert in_docs == normal_docs

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_ne(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"meta_field": {"$ne": "test-1"}})
        assert all("test-1" != d.meta.get("meta_field", None) for d in retrieved_docs)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_nin(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(
            filters={"meta_field": {"$nin": ["test-1", "test-2", "n.a."]}}
        )
        assert {"test-1", "test-2"}.isdisjoint({d.meta.get("meta_field", None) for d in retrieved_docs})

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_gt(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$gt": 3.0}})
        assert all(d.meta["numeric_field"] > 3.0 for d in retrieved_docs)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_gte(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$gte": 3.0}})
        assert all(d.meta["numeric_field"] >= 3.0 for d in retrieved_docs)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_lt(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$lt": 3.0}})
        assert all(d.meta["numeric_field"] < 3.0 for d in retrieved_docs)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_lte(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.get_all_documents(filters={"numeric_field": {"$lte": 3.0}})
        assert all(d.meta["numeric_field"] <= 3.0 for d in retrieved_docs)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_compound_dates(self, doc_store_with_docs: PineconeDocumentStore):
        filters = {"date": {"$lte": "2020-12-31", "$gte": "2019-01-01"}}

        with pytest.raises(FilterError, match=r"Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_compound_dates_and_other_field_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "name": ["file_5.txt", "file_3.txt"],
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters_simplified)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
    def test_get_all_documents_extended_filter_compound_dates_and_or_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "date": {"$lte": "2020-12-31", "$gte": "2019-01-01"},
            "$or": {"name": ["file_5.txt", "file_3.txt"], "numeric_field": {"$lte": 5.0}},
        }

        with pytest.raises(FilterError, match="Comparison value for '\$[l|g]te' operation must be a float or int."):
            doc_store_with_docs.get_all_documents(filters=filters_simplified)

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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

    @pytest.mark.pinecone
    # NOTE: Pinecone does not support dates, so it can't do lte or gte on date fields. When a new release introduces this feature,
    # the entire family of test_get_all_documents_extended_filter_* tests will become identical to the one present in the
    # base document store suite, and can be removed from here.
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
