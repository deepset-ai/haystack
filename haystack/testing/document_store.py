# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import random
from datetime import datetime

from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.errors import FilterError
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pytest'") as pytest_import:
    import pytest


def _random_embeddings(n):
    return [random.random() for _ in range(n)]


# These are random embedding that are used to test filters.
# We declare them here as they're used both in the `filterable_docs` fixture
# and the body of several `filter_documents` tests.
TEST_EMBEDDING_1 = _random_embeddings(768)
TEST_EMBEDDING_2 = _random_embeddings(768)


class AssertDocumentsEqualMixin:
    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
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

    @staticmethod
    def test_count_empty(document_store: DocumentStore):
        """Test count is zero for an empty document store"""
        assert document_store.count_documents() == 0

    @staticmethod
    def test_count_not_empty(document_store: DocumentStore):
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
    for these tests to work correctly.
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

    @staticmethod
    def test_write_documents_duplicate_skip(document_store: DocumentStore):
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

    @staticmethod
    def test_write_documents_invalid_input(document_store: DocumentStore):
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

    @staticmethod
    def test_delete_documents(document_store: DocumentStore):
        """Test delete_documents() normal behaviour."""
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents([doc.id])
        assert document_store.count_documents() == 0

    @staticmethod
    def test_delete_documents_empty_document_store(document_store: DocumentStore):
        """Test delete_documents() doesn't fail when called using an empty Document Store."""
        document_store.delete_documents(["non_existing_id"])

    @staticmethod
    def test_delete_documents_non_existing_document(document_store: DocumentStore):
        """Test delete_documents() doesn't delete any Document when called with non-existing id."""
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents(["non_existing_id"])

        # No Document has been deleted
        assert document_store.count_documents() == 1


def create_filterable_docs() -> list[Document]:
    """
    Create a list of filterable documents to be used in the filterable_docs fixture.
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
    return documents


class FilterableDocsFixtureMixin:
    """
    Mixin class that adds a filterable_docs() fixture to a test class.
    """

    @pytest.fixture
    def filterable_docs(self) -> list[Document]:
        """Fixture that returns a list of Documents that can be used to test filtering."""
        return create_filterable_docs()


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

    @staticmethod
    def test_comparison_greater_than_with_string(document_store, filterable_docs):
        """Test filter_documents() with > comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": "1"})

    @staticmethod
    def test_comparison_greater_than_with_list(document_store, filterable_docs):
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

    @staticmethod
    def test_comparison_greater_than_equal_with_string(document_store, filterable_docs):
        """Test filter_documents() with >= comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": "1"})

    @staticmethod
    def test_comparison_greater_than_equal_with_list(document_store, filterable_docs):
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

    @staticmethod
    def test_comparison_less_than_with_string(document_store, filterable_docs):
        """Test filter_documents() with < comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": "1"})

    @staticmethod
    def test_comparison_less_than_with_list(document_store, filterable_docs):
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

    @staticmethod
    def test_comparison_less_than_equal_with_string(document_store, filterable_docs):
        """Test filter_documents() with <= comparator and string"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": "1"})

    @staticmethod
    def test_comparison_less_than_equal_with_list(document_store, filterable_docs):
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

    @staticmethod
    def test_comparison_in_with_with_non_list(document_store, filterable_docs):
        """Test filter_documents() with 'in' comparator and non-iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "in", "value": 9})

    @staticmethod
    def test_comparison_in_with_with_non_list_iterable(document_store, filterable_docs):
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

    @staticmethod
    def test_comparison_not_in_with_with_non_list(document_store, filterable_docs):
        """Test filter_documents() with 'not in' comparator and non-iterable"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents({"field": "meta.number", "operator": "not in", "value": 9})

    @staticmethod
    def test_comparison_not_in_with_with_non_list_iterable(document_store, filterable_docs):
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


class DeleteAllTest:
    """
    Tests for Document Store delete_all_documents().

    To use it create a custom test class and override the `document_store` fixture.
    Only mix in for stores that implement delete_all_documents.
    """

    @staticmethod
    def test_delete_all_documents(document_store: DocumentStore):
        """
        Test delete_all_documents() normal behaviour.

        This test verifies that delete_all_documents() removes all documents from the store
        and that the store remains functional after deletion.
        """
        docs = [Document(content="first doc", id="1"), Document(content="second doc", id="2")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents()  # type:ignore[attr-defined]
        assert document_store.count_documents() == 0

        new_doc = Document(content="new doc after delete all", id="3")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

    @staticmethod
    def test_delete_all_documents_empty_store(document_store: DocumentStore):
        """
        Test delete_all_documents() on an empty store.

        This should not raise an error and should leave the store empty.
        """
        assert document_store.count_documents() == 0
        document_store.delete_all_documents()  # type:ignore[attr-defined]
        assert document_store.count_documents() == 0

    @staticmethod
    def _delete_all_supports_recreate(document_store: DocumentStore) -> tuple[bool, str | None]:
        """
        Return (True, param_name) if delete_all_documents has recreate_index or recreate_collection, else (False, None).
        """
        sig = inspect.signature(document_store.delete_all_documents)  # type:ignore[attr-defined]
        if "recreate_index" in sig.parameters:
            return True, "recreate_index"
        if "recreate_collection" in sig.parameters:
            return True, "recreate_collection"
        return False, None

    @staticmethod
    def test_delete_all_documents_without_recreate_index(document_store: DocumentStore):
        """
        Test delete_all_documents() with recreate_index/recreate_collection=False when supported.

        Skipped if the store's delete_all_documents does not have recreate_index or recreate_collection.
        """
        supports, param_name = DeleteAllTest._delete_all_supports_recreate(document_store)
        if not supports:
            pytest.skip("delete_all_documents has no recreate_index or recreate_collection parameter")

        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents(**{param_name: False})  # type:ignore[attr-defined]
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

    @staticmethod
    def test_delete_all_documents_with_recreate_index(document_store: DocumentStore):
        """
        Test delete_all_documents() with recreate_index/recreate_collection=True when supported.

        Skipped if the store's delete_all_documents does not have recreate_index or recreate_collection.
        """
        supports, param_name = DeleteAllTest._delete_all_supports_recreate(document_store)
        if not supports:
            pytest.skip("delete_all_documents has no recreate_index or recreate_collection parameter")

        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        document_store.delete_all_documents(**{param_name: True})  # type:ignore[attr-defined]
        assert document_store.count_documents() == 0

        new_doc = Document(id="3", content="New document after delete all with recreate")
        document_store.write_documents([new_doc])
        assert document_store.count_documents() == 1

        retrieved = document_store.filter_documents()
        assert len(retrieved) == 1
        assert retrieved[0].content == "New document after delete all with recreate"


class DeleteByFilterTest:
    """
    Tests for Document Store delete_by_filter().
    """

    @staticmethod
    def test_delete_by_filter(document_store: DocumentStore):
        """Delete documents matching a filter and verify count and remaining docs."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
            Document(content="Doc 3", meta={"category": "Alpha"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # `delete_by_filter` is not part of the DocumentStore protocol
        sig = inspect.signature(document_store.delete_by_filter)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        deleted_count = document_store.delete_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Alpha"}, **params
        )
        assert deleted_count == 2
        assert document_store.count_documents() == 1

        remaining_docs = document_store.filter_documents()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "Beta"

    @staticmethod
    def test_delete_by_filter_no_matches(document_store: DocumentStore):
        """Delete with a filter that matches no documents returns 0 and leaves store unchanged."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        deleted_count = document_store.delete_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Gamma"}
        )
        assert deleted_count == 0
        assert document_store.count_documents() == 2

    @staticmethod
    def test_delete_by_filter_advanced_filters(document_store: DocumentStore):
        """Delete with AND/OR filter combinations and verify remaining documents."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "Alpha", "year": 2024, "status": "published"}),
            Document(content="Doc 3", meta={"category": "Beta", "year": 2023, "status": "draft"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # `delete_by_filter` is not part of the DocumentStore protocol
        sig = inspect.signature(document_store.delete_by_filter)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        deleted_count = document_store.delete_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "Alpha"},
                    {"field": "meta.year", "operator": "==", "value": 2023},
                ],
            },
            **params,
        )
        assert deleted_count == 1
        assert document_store.count_documents() == 2

        deleted_count = document_store.delete_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "Beta"},
                    {"field": "meta.status", "operator": "==", "value": "published"},
                ],
            },
            **params,
        )
        assert deleted_count == 2
        assert document_store.count_documents() == 0


class UpdateByFilterTest:
    """
    Tests for Document Store update_by_filter().
    """

    @staticmethod
    def test_update_by_filter(document_store: DocumentStore, filterable_docs: list[Document]):
        """Update documents matching a filter and verify count and meta changes."""
        document_store.write_documents(filterable_docs)
        expected_count = len([d for d in filterable_docs if d.meta.get("chapter") == "intro"])
        assert document_store.count_documents() == len(filterable_docs)

        # `update_by_filter` is not part of the DocumentStore protocol
        sig = inspect.signature(document_store.update_by_filter)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        updated_count = document_store.update_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"}, meta={"updated": True}, **params
        )
        assert updated_count == expected_count

        updated_docs = document_store.filter_documents(
            filters={"field": "meta.updated", "operator": "==", "value": True}
        )
        assert len(updated_docs) == expected_count
        for doc in updated_docs:
            assert doc.meta["chapter"] == "intro"
            assert doc.meta["updated"] is True

        not_updated_docs = document_store.filter_documents(
            filters={"field": "meta.chapter", "operator": "==", "value": "abstract"}
        )
        for doc in not_updated_docs:
            assert doc.meta.get("updated") is not True

    @staticmethod
    def test_update_by_filter_no_matches(document_store: DocumentStore, filterable_docs: list[Document]):
        """Update with a filter that matches no documents returns 0 and leaves store unchanged."""
        document_store.write_documents(filterable_docs)
        initial_count = len(filterable_docs)
        assert document_store.count_documents() == initial_count

        updated_count = document_store.update_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.chapter", "operator": "==", "value": "nonexistent_chapter"}, meta={"updated": True}
        )
        assert updated_count == 0
        assert document_store.count_documents() == initial_count

    @staticmethod
    def test_update_by_filter_multiple_fields(document_store: DocumentStore, filterable_docs: list[Document]):
        """Update matching documents with multiple meta fields and verify all are set."""
        document_store.write_documents(filterable_docs)
        expected_count = len([d for d in filterable_docs if d.meta.get("chapter") == "intro"])
        assert document_store.count_documents() == len(filterable_docs)

        # `update_by_filter` is not part of the DocumentStore protocol
        sig = inspect.signature(document_store.update_by_filter)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        updated_count = document_store.update_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"},
            meta={"updated": True, "extra_field": "set"},
            **params,
        )
        assert updated_count == expected_count

        updated_docs = document_store.filter_documents(
            filters={"field": "meta.extra_field", "operator": "==", "value": "set"}
        )
        assert len(updated_docs) == expected_count
        for doc in updated_docs:
            assert doc.meta["updated"] is True
            assert doc.meta["extra_field"] == "set"
            assert doc.meta["chapter"] == "intro"
            assert doc.meta.get("number") == 2

        not_updated_docs = document_store.filter_documents(
            filters={"field": "meta.chapter", "operator": "==", "value": "abstract"}
        )
        for doc in not_updated_docs:
            assert doc.meta.get("extra_field") != "set"

    @staticmethod
    def test_update_by_filter_advanced_filters(document_store: DocumentStore):
        """Update with AND/OR filter combinations and verify updated documents."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "Alpha", "year": 2024, "status": "draft"}),
            Document(content="Doc 3", meta={"category": "Beta", "year": 2023, "status": "draft"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # `update_by_filter` is not part of the DocumentStore protocol
        sig = inspect.signature(document_store.update_by_filter)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        updated_count = document_store.update_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "Alpha"},
                    {"field": "meta.year", "operator": "==", "value": 2023},
                ],
            },
            meta={"status": "published"},
            **params,
        )
        assert updated_count == 1

        published_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 1
        assert published_docs[0].meta["category"] == "Alpha"
        assert published_docs[0].meta["year"] == 2023

        updated_count = document_store.update_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "Beta"},
                    {"field": "meta.year", "operator": "==", "value": 2024},
                ],
            },
            meta={"featured": True},
            **params,
        )
        assert updated_count == 2

        featured_docs = document_store.filter_documents(
            filters={"field": "meta.featured", "operator": "==", "value": True}
        )
        assert len(featured_docs) == 2


class UpdateByFilterAsyncTest:
    """
    Tests for Document Store update_by_filter_async().

    Only mix in for stores that implement update_by_filter_async.
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_update_by_filter_async(document_store: DocumentStore, filterable_docs: list[Document]):
        """Update documents matching a filter asynchronously and verify count and meta changes."""
        document_store.write_documents(filterable_docs)
        expected_count = len([d for d in filterable_docs if d.meta.get("chapter") == "intro"])
        assert document_store.count_documents() == len(filterable_docs)

        sig = inspect.signature(document_store.update_by_filter_async)  # type:ignore[attr-defined]
        params = {"refresh": True} if "refresh" in sig.parameters else {}
        updated_count = await document_store.update_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"}, meta={"updated": True}, **params
        )
        assert updated_count == expected_count

        updated_docs = document_store.filter_documents(
            filters={"field": "meta.updated", "operator": "==", "value": True}
        )
        assert len(updated_docs) == expected_count
        for doc in updated_docs:
            assert doc.meta["chapter"] == "intro"
            assert doc.meta["updated"] is True

        not_updated_docs = document_store.filter_documents(
            filters={"field": "meta.chapter", "operator": "==", "value": "abstract"}
        )
        for doc in not_updated_docs:
            assert doc.meta.get("updated") is not True


class CountDocumentsByFilterTest:
    """
    Tests for Document Store count_documents_by_filter().

    Only mix in for stores that implement count_documents_by_filter.
    """

    @staticmethod
    def test_count_documents_by_filter_simple(document_store: DocumentStore):
        """Test count_documents_by_filter() with a simple equality filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 3

        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert count == 1

    @staticmethod
    def test_count_documents_by_filter_compound(document_store: DocumentStore):
        """Test count_documents_by_filter() with AND filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count == 2

    @staticmethod
    def test_count_documents_by_filter_no_matches(document_store: DocumentStore):
        """Test count_documents_by_filter() when filter matches no documents."""
        docs = [Document(content="Doc 1", meta={"category": "A"}), Document(content="Doc 2", meta={"category": "B"})]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Z"}
        )
        assert count == 0

    @staticmethod
    def test_count_documents_by_filter_empty_collection(document_store: DocumentStore):
        """Test count_documents_by_filter() on an empty store."""
        assert document_store.count_documents() == 0

        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 0


class CountDocumentsByFilterAsyncTest:
    """
    Tests for Document Store count_documents_by_filter_async().

    Only mix in for stores that implement count_documents_by_filter_async.
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_simple(document_store: DocumentStore):
        """Test count_documents_by_filter_async() with a simple equality filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count = await document_store.count_documents_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 3

        count = await document_store.count_documents_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert count == 1

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_compound(document_store: DocumentStore):
        """Test count_documents_by_filter_async() with AND filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active"}),
            Document(content="Doc 2", meta={"category": "B", "status": "active"}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive"}),
            Document(content="Doc 4", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        count = await document_store.count_documents_by_filter_async(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.status", "operator": "==", "value": "active"},
                ],
            }
        )
        assert count == 2

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_no_matches(document_store: DocumentStore):
        """Test count_documents_by_filter_async() when filter matches no documents."""
        docs = [Document(content="Doc 1", meta={"category": "A"}), Document(content="Doc 2", meta={"category": "B"})]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        count = await document_store.count_documents_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Z"}
        )
        assert count == 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_empty_collection(document_store: DocumentStore):
        """Test count_documents_by_filter_async() on an empty store."""
        assert document_store.count_documents() == 0

        count = await document_store.count_documents_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 0


class CountUniqueMetadataByFilterTest:
    """
    Tests for Document Store count_unique_metadata_by_filter().

    Only mix in for stores that implement count_unique_metadata_by_filter.
    """

    @staticmethod
    def test_count_unique_metadata_by_filter_all_documents(document_store: DocumentStore):
        """Test count_unique_metadata_by_filter() with no filter returns distinct counts for all docs."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

        counts = document_store.count_unique_metadata_by_filter(  # type:ignore[attr-defined]
            filters={}, metadata_fields=["category", "status", "priority"]
        )
        assert counts["category"] == 3
        assert counts["status"] == 2
        assert counts["priority"] == 3

    @staticmethod
    def test_count_unique_metadata_by_filter_with_filter(document_store: DocumentStore):
        """Test count_unique_metadata_by_filter() with a filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        counts = document_store.count_unique_metadata_by_filter(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}, metadata_fields=["status", "priority"]
        )
        assert counts["status"] == 2
        assert counts["priority"] == 2

    @staticmethod
    def test_count_unique_metadata_by_filter_with_multiple_filters(document_store: DocumentStore):
        """Test counting with multiple filters"""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023}),
            Document(content="Doc 4", meta={"category": "B", "year": 2024}),
        ]
        document_store.write_documents(docs)
        count = document_store.count_documents_by_filter(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "B"},
                    {"field": "meta.year", "operator": "==", "value": 2023},
                ],
            }
        )
        assert count == 1


class CountUniqueMetadataByFilterAsyncTest:
    """
    Tests for Document Store count_unique_metadata_by_filter_async().

    Only mix in for stores that implement count_unique_metadata_by_filter_async.
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_all_documents(document_store: DocumentStore):
        """Test count_unique_metadata_by_filter_async() with no filter returns distinct counts for all docs."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
            Document(content="Doc 5", meta={"category": "C", "status": "active", "priority": 2}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

        counts = await document_store.count_unique_metadata_by_filter_async(  # type:ignore[attr-defined]
            filters={}, metadata_fields=["category", "status", "priority"]
        )
        assert counts["category"] == 3
        assert counts["status"] == 2
        assert counts["priority"] == 3

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_with_filter(document_store: DocumentStore):
        """Test count_unique_metadata_by_filter_async() with a filter."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "active", "priority": 2}),
            Document(content="Doc 3", meta={"category": "A", "status": "inactive", "priority": 1}),
            Document(content="Doc 4", meta={"category": "A", "status": "active", "priority": 3}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        counts = await document_store.count_unique_metadata_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "A"}, metadata_fields=["status", "priority"]
        )
        assert counts["status"] == 2
        assert counts["priority"] == 2

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async_with_multiple_filters(document_store: DocumentStore):
        """Test counting unique metadata asynchronously with multiple filters."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "year": 2023}),
            Document(content="Doc 2", meta={"category": "A", "year": 2024}),
            Document(content="Doc 3", meta={"category": "B", "year": 2023}),
            Document(content="Doc 4", meta={"category": "B", "year": 2024}),
        ]
        document_store.write_documents(docs)

        counts = await document_store.count_unique_metadata_by_filter_async(  # type:ignore[attr-defined]
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "B"},
                    {"field": "meta.year", "operator": "==", "value": 2023},
                ],
            },
            metadata_fields=["category", "year"],
        )
        assert counts == {"category": 1, "year": 1}


class GetMetadataFieldsInfoTest:
    """
    Tests for Document Store get_metadata_fields_info().

    Only mix in for stores that implement get_metadata_fields_info.
    """

    @staticmethod
    def test_get_metadata_fields_info(document_store: DocumentStore):
        """Test get_metadata_fields_info() returns field names and types after writing documents."""
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1}),
            Document(content="Doc 2", meta={"category": "B", "status": "inactive", "rating": 0.5}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        fields_info = document_store.get_metadata_fields_info()  # type:ignore[attr-defined]

        assert "category" in fields_info
        assert "status" in fields_info
        assert "priority" in fields_info
        assert "rating" in fields_info
        for field_name, info in fields_info.items():  # noqa: B007, PERF102
            assert isinstance(info, dict)
            assert "type" in info

    @staticmethod
    def test_get_metadata_fields_info_empty_collection(document_store: DocumentStore):
        """Test get_metadata_fields_info() on an empty store."""
        assert document_store.count_documents() == 0

        fields_info = document_store.get_metadata_fields_info()  # type:ignore[attr-defined]
        assert fields_info == {}


class GetMetadataFieldMinMaxTest:
    """
    Tests for Document Store get_metadata_field_min_max().

    Only mix in for stores that implement get_metadata_field_min_max.
    """

    @staticmethod
    def test_get_metadata_field_min_max_numeric(document_store: DocumentStore):
        """Test get_metadata_field_min_max() with integer field."""
        docs = [
            Document(content="Doc 1", meta={"priority": 1}),
            Document(content="Doc 2", meta={"priority": 5}),
            Document(content="Doc 3", meta={"priority": 3}),
            Document(content="Doc 4", meta={"priority": 10}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 4

        result = document_store.get_metadata_field_min_max("priority")  # type:ignore[attr-defined]
        assert result["min"] == 1
        assert result["max"] == 10

    @staticmethod
    def test_get_metadata_field_min_max_float(document_store: DocumentStore):
        """Test get_metadata_field_min_max() with float field."""
        docs = [
            Document(content="Doc 1", meta={"rating": 0.6}),
            Document(content="Doc 2", meta={"rating": 0.95}),
            Document(content="Doc 3", meta={"rating": 0.8}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        result = document_store.get_metadata_field_min_max("rating")  # type:ignore[attr-defined]

        assert result["min"] == pytest.approx(0.6)
        assert result["max"] == pytest.approx(0.95)

    @staticmethod
    def test_get_metadata_field_min_max_single_value(document_store: DocumentStore):
        """Test get_metadata_field_min_max() when field has only one value."""
        docs = [Document(content="Doc 1", meta={"priority": 42})]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 1

        result = document_store.get_metadata_field_min_max("priority")  # type:ignore[attr-defined]
        assert result["min"] == 42
        assert result["max"] == 42

    @staticmethod
    def test_get_metadata_field_min_max_empty_collection(document_store: DocumentStore):
        """Test get_metadata_field_min_max() on an empty store."""
        assert document_store.count_documents() == 0

        result = document_store.get_metadata_field_min_max("priority")  # type:ignore[attr-defined]
        assert result["min"] is None
        assert result["max"] is None

    @staticmethod
    def test_get_metadata_field_min_max_meta_prefix(document_store: DocumentStore):
        """Test get_metadata_field_min_max() with field names that include 'meta.' prefix."""
        docs = [
            Document(content="Doc 1", meta={"priority": 1, "age": 10}),
            Document(content="Doc 2", meta={"priority": 5, "age": 20}),
            Document(content="Doc 3", meta={"priority": 3, "age": 15}),
            Document(content="Doc 4", meta={"priority": 10, "age": 5}),
            Document(content="Doc 6", meta={"rating": 10.5}),
            Document(content="Doc 7", meta={"rating": 20.3}),
            Document(content="Doc 8", meta={"rating": 15.7}),
            Document(content="Doc 9", meta={"rating": 5.2}),
        ]
        document_store.write_documents(docs)

        min_max_priority = document_store.get_metadata_field_min_max("meta.priority")  # type:ignore[attr-defined]
        assert min_max_priority["min"] == 1
        assert min_max_priority["max"] == 10

        # Test with float values and "meta." prefix
        min_max_score = document_store.get_metadata_field_min_max("meta.rating")  # type:ignore[attr-defined]
        assert min_max_score["min"] == pytest.approx(5.2)
        assert min_max_score["max"] == pytest.approx(20.3)


class GetMetadataFieldUniqueValuesTest:
    """
    Tests for Document Store get_metadata_field_unique_values().

    Only mix in for stores that implement get_metadata_field_unique_values.
    Expects the method to return (values_list, total_count) or (values_list, pagination_key).
    """

    @staticmethod
    def test_get_metadata_field_unique_values_basic(document_store: DocumentStore):
        """Test get_metadata_field_unique_values() returns unique values and total count."""
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
            Document(content="Doc 4", meta={"category": "C"}),
            Document(content="Doc 5", meta={"category": "B"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

        sig = inspect.signature(document_store.get_metadata_field_unique_values)  # type:ignore[attr-defined]
        params: dict = {}
        if "search_term" in sig.parameters:
            params["search_term"] = None
        if "from_" in sig.parameters:
            params["from_"] = 0
        elif "offset" in sig.parameters:
            params["offset"] = 0
        if "size" in sig.parameters:
            params["size"] = 10
        elif "limit" in sig.parameters:
            params["limit"] = 10

        result = document_store.get_metadata_field_unique_values("category", **params)  # type:ignore[attr-defined]

        values = result[0] if isinstance(result, tuple) else result
        assert isinstance(values, list)
        assert set(values) == {"A", "B", "C"}
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], int):
            assert result[1] == 3


class DocumentStoreBaseTests(CountDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest, WriteDocumentsTest):
    @pytest.fixture
    def document_store(self) -> DocumentStore:
        """Base fixture, to be reimplemented when deriving from DocumentStoreBaseTests"""
        raise NotImplementedError()


class DocumentStoreBaseExtendedTests(DocumentStoreBaseTests, DeleteAllTest, DeleteByFilterTest, UpdateByFilterTest):
    """
    Extended tests for Document Stores.

    Besides the base tests, it also tests for:
    - delete_all_documents()
    - delete_by_filter()
    - update_by_filter()
    """

    @pytest.fixture
    def document_store(self) -> DocumentStore:
        """Base fixture, to be reimplemented when deriving from DocumentStoreBaseTests"""
        raise NotImplementedError()
