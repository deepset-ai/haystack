# pylint: disable=too-many-public-methods
from typing import List
import random

import pytest
import pandas as pd

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores import DocumentStore, DuplicatePolicy
from haystack.preview.document_stores.errors import DuplicateDocumentError
from haystack.preview.errors import FilterError


def _random_embeddings(n):
    return [random.random() for _ in range(n)]


# These are random embedding that are used to test filters.
# We declare them here as they're used both in the `filterable_docs` fixture
# and the body of several `filter_documents` tests.
TEST_EMBEDDING_1 = _random_embeddings(768)
TEST_EMBEDDING_2 = _random_embeddings(768)


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

    @pytest.mark.unit
    def test_count_empty(self, document_store: DocumentStore):
        assert document_store.count_documents() == 0

    @pytest.mark.unit
    def test_count_not_empty(self, document_store: DocumentStore):
        document_store.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert document_store.count_documents() == 3


class WriteDocumentsTest:
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

    @pytest.mark.unit
    def test_write_documents(self, document_store: DocumentStore):
        """
        Test write_documents() normal behaviour.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        assert document_store.filter_documents() == [doc]

    @pytest.mark.unit
    def test_write_documents_duplicate_fail(self, document_store: DocumentStore):
        """
        Test write_documents() fails when trying to write Document with same id
        using DuplicatePolicy.FAIL.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        assert document_store.filter_documents() == [doc]

    @pytest.mark.unit
    def test_write_documents_duplicate_skip(self, document_store: DocumentStore):
        """
        Test write_documents() skips Document when trying to write one with same id
        using DuplicatePolicy.SKIP.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        assert document_store.write_documents(documents=[doc], policy=DuplicatePolicy.SKIP) == 0

    @pytest.mark.unit
    def test_write_documents_duplicate_overwrite(self, document_store: DocumentStore):
        """
        Test write_documents() overwrites stored Document when trying to write one with same id
        using DuplicatePolicy.OVERWRITE.
        """
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2]) == 1
        assert document_store.filter_documents() == [doc2]
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        assert document_store.filter_documents() == [doc1]

    @pytest.mark.unit
    def test_write_documents_invalid_input(self, document_store: DocumentStore):
        """
        Test write_documents() fails when providing unexpected input.
        """
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

    @pytest.mark.unit
    def test_delete_documents(self, document_store: DocumentStore):
        """
        Test delete_documents() normal behaviour.
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        document_store.delete_documents([doc.id])
        assert document_store.count_documents() == 0

    @pytest.mark.unit
    def test_delete_documents_empty_document_store(self, document_store: DocumentStore):
        """
        Test delete_documents() doesn't fail when called using an empty Document Store.
        """
        document_store.delete_documents(["non_existing_id"])

    @pytest.mark.unit
    def test_delete_documents_non_existing_document(self, document_store: DocumentStore):
        """
        Test delete_documents() doesn't delete any Document when called with non existing id.
        """
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
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={"name": f"name_{i}", "page": "100", "chapter": "intro", "number": 2},
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={"name": f"name_{i}", "page": "123", "chapter": "abstract", "number": -2},
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={"name": f"name_{i}", "page": "90", "chapter": "conclusion", "number": -10},
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


class LegacyFilterDocumentsInvalidFiltersTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_incorrect_filter_type(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters="something odd")  # type: ignore

    @pytest.mark.unit
    def test_incorrect_filter_nesting(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"number": {"page": "100"}})

    @pytest.mark.unit
    def test_deeper_incorrect_filter_nesting(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"number": {"page": {"chapter": "intro"}}})


class LegacyFilterDocumentsEqualTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_filter_document_content(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"content": "A Foo Document 1"})
        assert result == [doc for doc in filterable_docs if doc.content == "A Foo Document 1"]

    @pytest.mark.unit
    def test_filter_simple_metadata_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": "100"})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") == "100"]

    @pytest.mark.unit
    def test_filter_document_dataframe(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        assert result == [
            doc for doc in filterable_docs if doc.dataframe is not None and doc.dataframe.equals(pd.DataFrame([1]))
        ]

    @pytest.mark.unit
    def test_eq_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$eq": "100"}})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") == "100"]

    @pytest.mark.unit
    def test_eq_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": "100"})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") == "100"]

    @pytest.mark.unit
    def test_eq_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        assert result == [
            doc
            for doc in filterable_docs
            if isinstance(doc.dataframe, pd.DataFrame) and doc.dataframe.equals(pd.DataFrame([1]))
        ]

    @pytest.mark.unit
    def test_eq_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        embedding = [0.0] * 768
        result = document_store.filter_documents(filters={"embedding": embedding})
        assert result == [doc for doc in filterable_docs if embedding == doc.embedding]


class LegacyFilterDocumentsNotEqualTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_ne_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$ne": "100"}})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") != "100"]

    @pytest.mark.unit
    def test_ne_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": {"$ne": pd.DataFrame([1])}})
        assert result == [
            doc
            for doc in filterable_docs
            if not isinstance(doc.dataframe, pd.DataFrame) or not doc.dataframe.equals(pd.DataFrame([1]))
        ]

    @pytest.mark.unit
    def test_ne_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"embedding": {"$ne": TEST_EMBEDDING_1}})
        assert result == [doc for doc in filterable_docs if doc.embedding != TEST_EMBEDDING_1]


class LegacyFilterDocumentsInTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_filter_simple_list_single_element(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100"]})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") == "100"]

    @pytest.mark.unit
    def test_filter_simple_list_one_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100"]})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") in ["100"]]

    @pytest.mark.unit
    def test_filter_simple_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100", "123"]})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]

    @pytest.mark.unit
    def test_incorrect_filter_name(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    @pytest.mark.unit
    def test_incorrect_filter_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["nope"]})
        assert len(result) == 0

    @pytest.mark.unit
    def test_in_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$in": ["100", "123", "n.a."]}})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]

    @pytest.mark.unit
    def test_in_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": ["100", "123", "n.a."]})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]

    @pytest.mark.unit
    def test_in_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"dataframe": {"$in": [pd.DataFrame([1]), pd.DataFrame([2])]}})
        assert result == [
            doc
            for doc in filterable_docs
            if isinstance(doc.dataframe, pd.DataFrame)
            and (doc.dataframe.equals(pd.DataFrame([1])) or doc.dataframe.equals(pd.DataFrame([2])))
        ]

    @pytest.mark.unit
    def test_in_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        embedding_zero = [0.0] * 768
        embedding_one = [1.0] * 768
        result = document_store.filter_documents(filters={"embedding": {"$in": [embedding_zero, embedding_one]}})
        assert result == [
            doc for doc in filterable_docs if (embedding_zero == doc.embedding or embedding_one == doc.embedding)
        ]


class LegacyFilterDocumentsNotInTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_nin_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}}
        )
        assert result == [
            doc
            for doc in filterable_docs
            if not isinstance(doc.dataframe, pd.DataFrame)
            or (not doc.dataframe.equals(pd.DataFrame([1])) and not doc.dataframe.equals(pd.DataFrame([0])))
        ]

    @pytest.mark.unit
    def test_nin_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"embedding": {"$nin": [TEST_EMBEDDING_1, TEST_EMBEDDING_2]}})
        assert result == [doc for doc in filterable_docs if doc.embedding not in [TEST_EMBEDDING_1, TEST_EMBEDDING_2]]

    @pytest.mark.unit
    def test_nin_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
        assert result == [doc for doc in filterable_docs if doc.meta.get("page") not in ["100", "123"]]


class LegacyFilterDocumentsGreaterThanTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_gt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$gt": 0.0}})
        assert result == [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] > 0]

    @pytest.mark.unit
    def test_gt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$gt": "100"}})

    @pytest.mark.unit
    def test_gt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_gt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$gt": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsGreaterThanEqualTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_gte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$gte": -2}})
        assert result == [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] >= -2]

    @pytest.mark.unit
    def test_gte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$gte": "100"}})

    @pytest.mark.unit
    def test_gte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_gte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$gte": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsLessThanTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_lt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lt": 0.0}})
        assert result == [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] < 0]

    @pytest.mark.unit
    def test_lt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$lt": "100"}})

    @pytest.mark.unit
    def test_lt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_lt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$lt": TEST_EMBEDDING_2}})


class LegacyFilterDocumentsLessThanEqualTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_lte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0}})
        assert result == [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] <= 2.0]

    @pytest.mark.unit
    def test_lte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$lte": "100"}})

    @pytest.mark.unit
    def test_lte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"dataframe": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})

    @pytest.mark.unit
    def test_lte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"embedding": {"$lte": TEST_EMBEDDING_1}})


class LegacyFilterDocumentsSimpleLogicalTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_filter_simple_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        result = document_store.filter_documents(filters=filters)
        assert result == [
            doc
            for doc in filterable_docs
            if (("number" in doc.meta and doc.meta["number"] < 1) or doc.meta.get("name") in ["name_0", "name_1"])
        ]

    @pytest.mark.unit
    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0.0}})
        assert result == [
            doc
            for doc in filterable_docs
            if "number" in doc.meta and doc.meta["number"] >= 0.0 and doc.meta["number"] <= 2.0
        ]

    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}})
        assert result == [
            doc
            for doc in filterable_docs
            if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
        ]

    @pytest.mark.unit
    def test_filter_simple_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0}})
        assert result == [
            doc
            for doc in filterable_docs
            if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
        ]


class LegacyFilterDocumentsNestedLogicalTest(FilterableDocsFixtureMixin):
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

    @pytest.mark.unit
    def test_filter_nested_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]}
        result = document_store.filter_documents(filters=filters_simplified)
        assert result == [
            doc
            for doc in filterable_docs
            if (
                "number" in doc.meta
                and doc.meta["number"] <= 2
                and doc.meta["number"] >= 0
                and doc.meta.get("name") in ["name_0", "name_1"]
            )
        ]

    @pytest.mark.unit
    def test_filter_nested_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}}
        result = document_store.filter_documents(filters=filters)
        assert result == [
            doc
            for doc in filterable_docs
            if (doc.meta.get("name") in ["name_0", "name_1"] or ("number" in doc.meta and doc.meta["number"] < 1))
        ]

    @pytest.mark.unit
    def test_filter_nested_and_or_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {"page": {"$eq": "123"}, "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        }
        result = document_store.filter_documents(filters=filters_simplified)
        assert result == [
            doc
            for doc in filterable_docs
            if (
                doc.meta.get("page") in ["123"]
                and (doc.meta.get("name") in ["name_0", "name_1"] or ("number" in doc.meta and doc.meta["number"] < 1))
            )
        ]

    @pytest.mark.unit
    def test_filter_nested_and_or_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "page": {"$eq": "123"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = document_store.filter_documents(filters=filters_simplified)
        assert result == [
            doc
            for doc in filterable_docs
            if (
                doc.meta.get("page") in ["123"]
                and (doc.meta.get("name") in ["name_0", "name_1"] or ("number" in doc.meta and doc.meta["number"] < 1))
            )
        ]

    @pytest.mark.unit
    def test_filter_nested_or_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        assert result == [
            doc
            for doc in filterable_docs
            if (
                ("number" in doc.meta and doc.meta["number"] < 1)
                or (
                    doc.meta.get("name") in ["name_0", "name_1"]
                    and ("chapter" in doc.meta and doc.meta["chapter"] != "intro")
                )
            )
        ]

    @pytest.mark.unit
    def test_filter_nested_multiple_identical_operators_same_level(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": "100"}},
                {"$and": {"chapter": {"$in": ["intro", "abstract"]}, "page": "123"}},
            ]
        }
        result = document_store.filter_documents(filters=filters)
        assert result == [
            doc
            for doc in filterable_docs
            if (
                (doc.meta.get("name") in ["name_0", "name_1"] and doc.meta.get("page") == "100")
                or (doc.meta.get("chapter") in ["intro", "abstract"] and doc.meta.get("page") == "123")
            )
        ]


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

    @pytest.mark.unit
    def test_no_filter_empty(self, document_store: DocumentStore):
        assert document_store.filter_documents() == []
        assert document_store.filter_documents(filters={}) == []

    @pytest.mark.unit
    def test_no_filter_not_empty(self, document_store: DocumentStore):
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        assert document_store.filter_documents() == docs
        assert document_store.filter_documents(filters={}) == docs


class DocumentStoreBaseTests(
    CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest, LegacyFilterDocumentsTest
):  # pylint: disable=too-many-ancestors
    @pytest.fixture
    def document_store(self) -> DocumentStore:
        raise NotImplementedError()
