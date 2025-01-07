# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from datetime import datetime
from typing import List

import pandas as pd

from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.errors import FilterError
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pytest'") as pytest_import:
    import pytest


def _random_embeddings(n):
    return [random.random() for _ in range(n)]


# pylint: disable=too-many-public-methods


# These are random embedding that are used to test filters.
# We declare them here as they're used both in the `filterable_docs` fixture
# and the body of several `filter_documents` tests.
TEST_EMBEDDING_1 = _random_embeddings(768)
TEST_EMBEDDING_2 = _random_embeddings(768)


class AssertDocumentsEqualMixin:
    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.

        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method. This can happen for example when the Document Store sets
        a score to returned Documents. Since we can't know what the score will be, we can't compare
        the Documents reliably.
        """
        assert received == expected


class CountDocumentsTest:
    """
    Utility class to test a Document Store `count_documents` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(CountDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_count_empty(self, document_store: DocumentStore):
        """Test count is zero for an empty document store"""
        assert document_store.count_documents() == 0

    def test_count_not_empty(self, document_store: DocumentStore):
        """Test count is greater than zero if the document store contains documents"""
        document_store.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert document_store.count_documents() == 3


class WriteDocumentsTest(AssertDocumentsEqualMixin):
    """
    Utility class to test a Document Store `write_documents` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    The Document Store `filter_documents` method must be at least partly implemented to return all stored Documents
    for this tests to work correctly.
    Example usage:

    ```python
    class MyDocumentStoreTest(WriteDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_write_documents(self, document_store: DocumentStore):
        """
        Test write_documents() default behaviour.
        """
        msg = (
            "Default write_documents() behaviour depends on the Document Store implementation, "
            "as we don't enforce a default behaviour when no policy is set. "
            "Override this test in your custom test class."
        )
        raise NotImplementedError(msg)

    def test_write_documents_duplicate_fail(self, document_store: DocumentStore):
        """Test write_documents() fails when writing documents with same id and `DuplicatePolicy.FAIL`."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc], policy=DuplicatePolicy.FAIL) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    def test_write_documents_duplicate_skip(self, document_store: DocumentStore):
        """Test write_documents() skips writing when using DuplicatePolicy.SKIP."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc], policy=DuplicatePolicy.SKIP) == 1
        assert document_store.write_documents(documents=[doc], policy=DuplicatePolicy.SKIP) == 0

    def test_write_documents_duplicate_overwrite(self, document_store: DocumentStore):
        """Test write_documents() overwrites when using DuplicatePolicy.OVERWRITE."""
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])

    def test_write_documents_invalid_input(self, document_store: DocumentStore):
        """Test write_documents() fails when providing unexpected input."""
        with pytest.raises(ValueError):
            document_store.write_documents(["not a document for sure"])  # type: ignore
        with pytest.raises(ValueError):
            document_store.write_documents("not a list actually")  # type: ignore


class DeleteDocumentsTest:
    """
    Utility class to test a Document Store `delete_documents` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    The Document Store `write_documents` and `count_documents` methods must be implemented for this tests to work
    correctly.
    Example usage:

    ```python
    class MyDocumentStoreTest(DeleteDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_delete_documents(self, document_store: DocumentStore):
        """Test delete_documents() normal behaviour."""
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents([doc.id])
        assert document_store.count_documents() == 0

    def test_delete_documents_empty_document_store(self, document_store: DocumentStore):
        """Test delete_documents() doesn't fail when called using an empty Document Store."""
        document_store.delete_documents(["non_existing_id"])

    def test_delete_documents_non_existing_document(self, document_store: DocumentStore):
        """Test delete_documents() doesn't delete any Document when called with non existing id."""
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents(["non_existing_id"])

        # No Document has been deleted
        assert document_store.count_documents() == 1


def create_filterable_docs(include_dataframe_docs: bool = False) -> List[Document]:
    """
    Create a list of filterable documents to be used in the filterable_docs and filterable_docs_with_dataframe fixtures.
    """

    documents = []
    for i in range(3):
        documents.append(
            Document(
                content=f"A Foo Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"A Bar Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "123",
                    "chapter": "abstract",
                    "number": -2,
                    "date": "1972-12-11T19:54:58",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"A Foobar Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "90",
                    "chapter": "conclusion",
                    "number": -10,
                    "date": "1989-11-09T17:53:00",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"Document {i} without embedding",
                meta={"name": f"name_{i}", "no_embedding": True, "chapter": "conclusion"},
            )
        )
        documents.append(
            Document(content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=TEST_EMBEDDING_1)
        )
        documents.append(
            Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=TEST_EMBEDDING_2)
        )

    if include_dataframe_docs:
        for i in range(3):
            documents.append(Document(dataframe=pd.DataFrame([i]), meta={"name": f"table_doc_{i}"}))

    return documents


class FilterableDocsFixtureMixin:
    """
    Mixin class that adds a filterable_docs() fixture to a test class.
    """

    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        """Fixture that returns a list of Documents that can be used to test filtering."""
        return create_filterable_docs(include_dataframe_docs=False)


class FilterDocumentsTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using different types of filters.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(FilterDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_no_filters(self, document_store):
        """Test filter_documents() with empty filters"""
        self.assert_documents_are_equal(document_store.filter_documents(), [])
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), [])
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        self.assert_documents_are_equal(document_store.filter_documents(), docs)
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), docs)

    # == comparator
    def test_comparison_equal(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": 100})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") == 100])

    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") is None])

    # != comparator
    def test_comparison_not_equal(self, document_store, filterable_docs):
        """Test filter_documents() with != comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "!=", "value": 100})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") != 100])

    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with != comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "!=", "value": None})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") is not None])

    # > comparator
    def test_comparison_greater_than(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": ">", "value": 0})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") is not None and d.meta["number"] > 0]
        )

    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.fromisoformat(d.meta["date"]) > datetime.fromisoformat("1972-12-11T19:54:58")
            ],
        )

    def test_comparison_greater_than_with_string(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": "1"})

    def test_comparison_greater_than_with_list(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and list"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": [1]})

    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": None})
        self.assert_documents_are_equal(result, [])

    # >= comparator
    def test_comparison_greater_than_equal(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": ">=", "value": 0})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") is not None and d.meta["number"] >= 0]
        )

    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">=", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.fromisoformat(d.meta["date"]) >= datetime.fromisoformat("1969-07-21T20:17:40")
            ],
        )

    def test_comparison_greater_than_equal_with_string(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": "1"})

    def test_comparison_greater_than_equal_with_list(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and list"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": [1]})

    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": None})
        self.assert_documents_are_equal(result, [])

    # < comparator
    def test_comparison_less_than(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "<", "value": 0})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") is not None and d.meta["number"] < 0]
        )

    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.fromisoformat(d.meta["date"]) < datetime.fromisoformat("1969-07-21T20:17:40")
            ],
        )

    def test_comparison_less_than_with_string(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": "1"})

    def test_comparison_less_than_with_list(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and list"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": [1]})

    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": None})
        self.assert_documents_are_equal(result, [])

    # <= comparator
    def test_comparison_less_than_equal(self, document_store, filterable_docs):
        """Test filter_documents() with <="""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "<=", "value": 0})
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") is not None and d.meta["number"] <= 0]
        )

    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<=", "value": "1969-07-21T20:17:40"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.fromisoformat(d.meta["date"]) <= datetime.fromisoformat("1969-07-21T20:17:40")
            ],
        )

    def test_comparison_less_than_equal_with_string(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": "1"})

    def test_comparison_less_than_equal_with_list(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and list"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": [1]})

    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and None"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": None})
        self.assert_documents_are_equal(result, [])

    # in comparator
    def test_comparison_in(self, document_store, filterable_docs):
        """Test filter_documents() with 'in' comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "in", "value": [10, -10]})
        assert len(result)
        expected = [d for d in filterable_docs if d.meta.get("number") is not None and d.meta["number"] in [10, -10]]
        self.assert_documents_are_equal(result, expected)

    def test_comparison_in_with_with_non_list(self, document_store, filterable_docs):
        """Test filter_documents() with 'in' comparator and non-iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "in", "value": 9})

    def test_comparison_in_with_with_non_list_iterable(self, document_store, filterable_docs):
        """Test filter_documents() with 'in' comparator and iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "in", "value": (10, 11)})

    # not in comparator
    def test_comparison_not_in(self, document_store, filterable_docs):
        """Test filter_documents() with 'not in' comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "not in", "value": [9, 10]})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") not in [9, 10]])

    def test_comparison_not_in_with_with_non_list(self, document_store, filterable_docs):
        """Test filter_documents() with 'not in' comparator and non-iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "not in", "value": 9})

    def test_comparison_not_in_with_with_non_list_iterable(self, document_store, filterable_docs):
        """Test filter_documents() with 'not in' comparator and iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "not in", "value": (10, 11)})

    # Logical operator
    def test_and_operator(self, document_store, filterable_docs):
        """Test filter_documents() with 'AND' operator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") == 100 and d.meta.get("name") == "name_0"]
        )

    def test_or_operator(self, document_store, filterable_docs):
        """Test filter_documents() with 'OR' operator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") == 100 or d.meta.get("name") == "name_0"]
        )

    def test_not_operator(self, document_store, filterable_docs):
        """Test filter_documents() with 'NOT' operator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if not (d.meta.get("number") == 100 and d.meta.get("name") == "name_0")]
        )

    # Malformed filters
    def test_missing_top_level_operator_key(self, document_store, filterable_docs):
        """Test filter_documents() with top-level operator"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"conditions": [{"field": "meta.name", "operator": "==", "value": "test"}]}
            )

    def test_missing_top_level_conditions_key(self, document_store, filterable_docs):
        """Test filter_documents() with missing top-level condition key"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"operator": "AND"})

    def test_missing_condition_field_key(self, document_store, filterable_docs):
        """Test filter_documents() with missing condition key"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"operator": "AND", "conditions": [{"operator": "==", "value": "test"}]}
            )

    def test_missing_condition_operator_key(self, document_store, filterable_docs):
        """Test filter_documents() with missing operator key"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"operator": "AND", "conditions": [{"field": "meta.name", "value": "test"}]}
            )

    def test_missing_condition_value_key(self, document_store, filterable_docs):
        """Test filter_documents() with missing condition value"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"operator": "AND", "conditions": [{"field": "meta.name", "operator": "=="}]}
            )


class FilterableDocsFixtureMixinWithDataframe:
    """
    Mixin class that adds a filterable_docs_with_dataframe() fixture to a test class, including dataframe documents.
    """

    @pytest.fixture
    def filterable_docs_with_dataframe(self) -> List[Document]:
        """Fixture that returns a list of Documents including dataframe documents."""
        documents = create_filterable_docs(include_dataframe_docs=True)

        return documents


class FilterDocumentsTestWithDataframe(AssertDocumentsEqualMixin, FilterableDocsFixtureMixinWithDataframe):
    """
    Utility class to test a Document Store `filter_documents` method specifically for DataFrame documents.
    """

    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with == comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        result = document_store.filter_documents(
            filters={"field": "dataframe", "operator": "==", "value": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs_with_dataframe
                if d.dataframe is not None and d.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with != comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        result = document_store.filter_documents(
            filters={"field": "dataframe", "operator": "!=", "value": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs_with_dataframe
                if d.dataframe is None or not d.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    def test_comparison_greater_than_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with > comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "dataframe", "operator": ">", "value": pd.DataFrame([1])})

    def test_comparison_greater_than_equal_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with >= comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"field": "dataframe", "operator": ">=", "value": pd.DataFrame([1])}
            )

    def test_comparison_less_than_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with < comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "dataframe", "operator": "<", "value": pd.DataFrame([1])})

    def test_comparison_less_than_equal_with_dataframe(self, document_store, filterable_docs_with_dataframe):
        """Test filter_documents() with <= comparator and dataframe"""
        document_store.write_documents(filterable_docs_with_dataframe)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"field": "dataframe", "operator": "<=", "value": pd.DataFrame([1])}
            )


class DocumentStoreBaseTests(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest):
    @pytest.fixture
    def document_store(self) -> DocumentStore:
        """Base fixture, to be reimplemented when deriving from DocumentStoreBaseTests"""
        raise NotImplementedError()
