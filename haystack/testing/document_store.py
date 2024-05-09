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
    The Document Store `write_documents` and `count_documents` methods must be implemented for this tests to work correctly.
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


class FilterableDocsFixtureMixin:
    """
    Mixin class that adds a filterable_docs() fixture to a test class.
    """

    @pytest.fixture
    def filterable_docs(self) -> List[Document]:
        """Fixture that returns a list of Documents that can be used to test filtering."""
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
            documents.append(Document(dataframe=pd.DataFrame([i]), meta={"name": f"table_doc_{i}"}))
            documents.append(
                Document(content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=TEST_EMBEDDING_1)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=TEST_EMBEDDING_2)
            )
        return documents


class LegacyFilterDocumentsInvalidFiltersTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using invalid legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsInvalidFiltersTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_incorrect_filter_type(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(ValueError):
            document_store.filter_documents(filters="something odd")  # type: ignore

    def test_incorrect_filter_nesting(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"number": {"page": "100"}})

    def test_deeper_incorrect_filter_nesting(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"number": {"page": {"chapter": "intro"}}})


class LegacyFilterDocumentsEqualTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using implicit and explicit '$eq' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_filter_document_content(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"content": "A Foo Document 1"})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.content == "A Foo Document 1"])

    def test_filter_simple_metadata_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": "100"})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_filter_document_dataframe(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.dataframe is not None and doc.dataframe.equals(pd.DataFrame([1]))],
        )

    def test_eq_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$eq": "100"}})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_eq_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": "100"})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_eq_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.dataframe, pd.DataFrame) and doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    def test_eq_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        embedding = [0.0] * 768
        result = document_store.filter_documents(filters={"embedding": embedding})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if embedding == doc.embedding])


class LegacyFilterDocumentsNotEqualTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$ne' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNotEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_ne_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$ne": "100"}})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") != "100"])

    def test_ne_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": {"$ne": pd.DataFrame([1])}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.dataframe, pd.DataFrame) or not doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    def test_ne_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"embedding": {"$ne": TEST_EMBEDDING_1}})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.embedding != TEST_EMBEDDING_1])


class LegacyFilterDocumentsInTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using implicit and explicit '$in' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsInTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_filter_simple_list_single_element(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100"]})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_filter_simple_list_one_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100"]})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100"]])

    def test_filter_simple_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100", "123"]})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    def test_incorrect_filter_name(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        self.assert_documents_are_equal(result, [])

    def test_incorrect_filter_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["nope"]})
        self.assert_documents_are_equal(result, [])

    def test_in_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$in": ["100", "123", "n.a."]}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    def test_in_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100", "123", "n.a."]})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    def test_in_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": {"$in": [pd.DataFrame([1]), pd.DataFrame([2])]}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.dataframe, pd.DataFrame)
                and (doc.dataframe.equals(pd.DataFrame([1])) or doc.dataframe.equals(pd.DataFrame([2])))
            ],
        )

    def test_in_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        embedding_zero = [0.0] * 768
        embedding_one = [1.0] * 768
        result = document_store.filter_documents(filters={"embedding": {"$in": [embedding_zero, embedding_one]}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if (embedding_zero == doc.embedding or embedding_one == doc.embedding)],
        )


class LegacyFilterDocumentsNotInTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$nin' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNotInTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_nin_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.dataframe, pd.DataFrame)
                or (not doc.dataframe.equals(pd.DataFrame([1])) and not doc.dataframe.equals(pd.DataFrame([0])))
            ],
        )

    def test_nin_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"embedding": {"$nin": [TEST_EMBEDDING_1, TEST_EMBEDDING_2]}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.embedding not in [TEST_EMBEDDING_1, TEST_EMBEDDING_2]]
        )

    def test_nin_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") not in ["100", "123"]]
        )


class LegacyFilterDocumentsGreaterThanTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$gt' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsGreaterThanTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_gt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$gt": 0.0}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] > 0]
        )

    def test_gt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$gt": "100"}})

    def test_gt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_gt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$gt": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsGreaterThanEqualTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$gte' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsGreaterThanEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_gte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$gte": -2}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] >= -2]
        )

    def test_gte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$gte": "100"}})

    def test_gte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_gte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$gte": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsLessThanTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$lt' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsLessThanTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_lt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lt": 0.0}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("number") is not None and doc.meta["number"] < 0]
        )

    def test_lt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$lt": "100"}})

    def test_lt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_lt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$lt": TEST_EMBEDDING_2}})


class LegacyFilterDocumentsLessThanEqualTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$lte' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsLessThanEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_lte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("number") is not None and doc.meta["number"] <= 2.0]
        )

    def test_lte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$lte": "100"}})

    def test_lte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    def test_lte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$lte": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsSimpleLogicalTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using logical '$and', '$or' and '$not' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsSimpleLogicalTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_filter_simple_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                or doc.meta.get("name") in ["name_0", "name_1"]
            ],
        )

    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0.0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] >= 0.0 and doc.meta["number"] <= 2.0
            ],
        )

    def test_filter_simple_explicit_and_with_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )

    def test_filter_simple_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )


class LegacyFilterDocumentsNestedLogicalTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using multiple nested logical '$and', '$or' and '$not' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNestedLogicalTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_filter_nested_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters_simplified = {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]}
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.meta
                    and doc.meta["number"] <= 2
                    and doc.meta["number"] >= 0
                    and doc.meta.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    def test_filter_nested_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}}
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("name") in ["name_0", "name_1"]
                    or (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                )
            ],
        )

    def test_filter_nested_and_or_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {"page": {"$eq": "123"}, "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_and_or_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "page": {"$eq": "123"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_or_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                    or (doc.meta.get("name") in ["name_0", "name_1"] and (doc.meta.get("chapter") != "intro"))
                )
            ],
        )

    def test_filter_nested_multiple_identical_operators_same_level(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        """"""  # noqa # pylint: disable=C0112
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": "100"}},
                {"$and": {"chapter": {"$in": ["intro", "abstract"]}, "page": "123"}},
            ]
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("name") in ["name_0", "name_1"] and doc.meta.get("page") == "100")
                    or (doc.meta.get("chapter") in ["intro", "abstract"] and doc.meta.get("page") == "123")
                )
            ],
        )


class LegacyFilterDocumentsTest(  # pylint: disable=too-many-ancestors
    LegacyFilterDocumentsInvalidFiltersTest,
    LegacyFilterDocumentsEqualTest,
    LegacyFilterDocumentsNotEqualTest,
    LegacyFilterDocumentsInTest,
    LegacyFilterDocumentsNotInTest,
    LegacyFilterDocumentsGreaterThanTest,
    LegacyFilterDocumentsGreaterThanEqualTest,
    LegacyFilterDocumentsLessThanTest,
    LegacyFilterDocumentsLessThanEqualTest,
    LegacyFilterDocumentsSimpleLogicalTest,
    LegacyFilterDocumentsNestedLogicalTest,
):
    """
    Utility class to test a Document Store `filter_documents` method using different types of legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    def test_no_filter_empty(self, document_store: DocumentStore):
        """"""  # noqa # pylint: disable=C0112
        assert document_store.filter_documents() == []
        assert document_store.filter_documents(filters={}) == []

    def test_no_filter_not_empty(self, document_store: DocumentStore):
        """"""  # noqa # pylint: disable=C0112
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        assert document_store.filter_documents() == docs
        assert document_store.filter_documents(filters={}) == docs


class FilterDocumentsTest(AssertDocumentsEqualMixin, FilterableDocsFixtureMixin):
    """
    Utility class to test a Document Store `filter_documents` method using different types of  filters.

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

    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with == comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"field": "dataframe", "operator": "==", "value": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.dataframe is not None and d.dataframe.equals(pd.DataFrame([1]))]
        )

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

    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with != comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"field": "dataframe", "operator": "!=", "value": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.dataframe is None or not d.dataframe.equals(pd.DataFrame([1]))]
        )

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

    def test_comparison_greater_than_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "dataframe", "operator": ">", "value": pd.DataFrame([1])})

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

    def test_comparison_greater_than_equal_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"field": "dataframe", "operator": ">=", "value": pd.DataFrame([1])}
            )

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

    def test_comparison_less_than_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "dataframe", "operator": "<", "value": pd.DataFrame([1])})

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

    def test_comparison_less_than_equal_with_dataframe(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and dataframe"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"field": "dataframe", "operator": "<=", "value": pd.DataFrame([1])}
            )

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


class DocumentStoreBaseTests(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest, FilterDocumentsTest):
    @pytest.fixture
    def document_store(self) -> DocumentStore:
        """Base fixture, to be reimplemented when deriving from DocumentStoreBaseTests"""
        raise NotImplementedError()
