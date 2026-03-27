# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Protocol

import pytest

from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.testing.document_store import AssertDocumentsEqualMixin


class AsyncDocumentStore(DocumentStore, Protocol):
    async def count_documents_async(self) -> int:
        """
        Returns the number of documents stored.
        """
        ...

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.
        """
        ...

    async def write_documents_async(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Writes Documents into the DocumentStore.
        """
        ...

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.
        """
        ...


class DeleteAllAsyncTest:
    """
    Tests for Document Store delete_all_documents_async().

    To use it create a custom test class and override the `document_store` fixture.
    Only mix in for stores that implement delete_all_documents_async.
    """

    @staticmethod
    def _delete_all_supports_recreate(document_store: AsyncDocumentStore) -> tuple[bool, str | None]:
        """
        Return (True, param_name) if delete_all_documents_async has recreate_index or recreate_collection.
        """
        sig = inspect.signature(document_store.delete_all_documents_async)  # type:ignore[attr-defined]
        if "recreate_index" in sig.parameters:
            return True, "recreate_index"
        if "recreate_collection" in sig.parameters:
            return True, "recreate_collection"
        return False, None

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_all_documents_async(document_store: AsyncDocumentStore):
        """
        Test delete_all_documents_async() normal behaviour.

        This test verifies that delete_all_documents_async() removes all documents from the store
        and that the store remains functional after deletion.
        """
        docs = [Document(content="first doc", id="1"), Document(content="second doc", id="2")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async()  # type:ignore[attr-defined]
        assert await document_store.count_documents_async() == 0

        new_doc = Document(content="new doc after delete all", id="3")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_all_documents_empty_store_async(document_store: AsyncDocumentStore):
        """
        Test delete_all_documents_async() on an empty store.

        This should not raise an error and should leave the store empty.
        """
        assert await document_store.count_documents_async() == 0
        await document_store.delete_all_documents_async()  # type:ignore[attr-defined]


class CountDocumentsAsyncTest:
    """
    Utility class to test a Document Store `count_documents_async` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(CountDocumentsAsyncTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_count_empty_async(document_store: AsyncDocumentStore):
        """Test count is zero for an empty document store."""
        assert await document_store.count_documents_async() == 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_all_documents_without_recreate_index_async(document_store: AsyncDocumentStore):
        """
        Test delete_all_documents_async() with recreate_index/recreate_collection=False when supported.

        Skipped if the store's delete_all_documents_async does not have recreate_index or recreate_collection.
        """
        supports, param_name = DeleteAllAsyncTest._delete_all_supports_recreate(document_store)
        if not supports or param_name is None:
            pytest.skip("delete_all_documents_async has no recreate_index or recreate_collection parameter")

        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(**{param_name: False})  # type:ignore[attr-defined]
        assert await document_store.count_documents_async() == 0

        new_doc = Document(id="3", content="New document after delete all")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_all_documents_with_recreate_index_async(document_store: AsyncDocumentStore):
        """
        Test delete_all_documents_async() with recreate_index/recreate_collection=True when supported.

        Skipped if the store's delete_all_documents_async does not have recreate_index or recreate_collection.
        """
        supports, param_name = DeleteAllAsyncTest._delete_all_supports_recreate(document_store)
        if not supports or param_name is None:
            pytest.skip("delete_all_documents_async has no recreate_index or recreate_collection parameter")

        docs = [Document(id="1", content="A first document"), Document(id="2", content="Second document")]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        await document_store.delete_all_documents_async(**{param_name: True})  # type:ignore[attr-defined]
        assert await document_store.count_documents_async() == 0

        new_doc = Document(id="3", content="New document after delete all with recreate")
        await document_store.write_documents_async([new_doc])
        assert await document_store.count_documents_async() == 1

        retrieved = await document_store.filter_documents_async()
        assert len(retrieved) == 1
        assert retrieved[0].content == "New document after delete all with recreate"


class DeleteByFilterAsyncTest:
    """
    Tests for Document Store delete_by_filter_async().
    """

    @staticmethod
    def _delete_by_filter_params(document_store: AsyncDocumentStore) -> dict[str, bool]:
        """
        Return optional parameters supported by delete_by_filter_async.
        """
        sig = inspect.signature(document_store.delete_by_filter_async)  # type:ignore[attr-defined]
        return {"refresh": True} if "refresh" in sig.parameters else {}

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_by_filter_async(document_store: AsyncDocumentStore):
        """Delete documents matching a filter and verify count and remaining docs."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
            Document(content="Doc 3", meta={"category": "Alpha"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        params = DeleteByFilterAsyncTest._delete_by_filter_params(document_store)
        deleted_count = await document_store.delete_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Alpha"}, **params
        )
        assert deleted_count == 2
        assert await document_store.count_documents_async() == 1

        remaining_docs = await document_store.filter_documents_async()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "Beta"

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_by_filter_no_matches_async(document_store: AsyncDocumentStore):
        """Delete with a filter that matches no documents returns 0 and leaves store unchanged."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 2

        params = DeleteByFilterAsyncTest._delete_by_filter_params(document_store)
        deleted_count = await document_store.delete_by_filter_async(  # type:ignore[attr-defined]
            filters={"field": "meta.category", "operator": "==", "value": "Gamma"}, **params
        )
        assert deleted_count == 0
        assert await document_store.count_documents_async() == 2

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_by_filter_advanced_filters_async(document_store: AsyncDocumentStore):
        """Delete with AND/OR filter combinations and verify remaining documents."""
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha", "year": 2023, "status": "draft"}),
            Document(content="Doc 2", meta={"category": "Alpha", "year": 2024, "status": "published"}),
            Document(content="Doc 3", meta={"category": "Beta", "year": 2023, "status": "draft"}),
        ]
        await document_store.write_documents_async(docs)
        assert await document_store.count_documents_async() == 3

        params = DeleteByFilterAsyncTest._delete_by_filter_params(document_store)
        deleted_count = await document_store.delete_by_filter_async(  # type:ignore[attr-defined]
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
        assert await document_store.count_documents_async() == 2

        deleted_count = await document_store.delete_by_filter_async(  # type:ignore[attr-defined]
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
        assert await document_store.count_documents_async() == 0

    async def test_count_not_empty_async(document_store: AsyncDocumentStore):
        """Test count is greater than zero if the document store contains documents."""
        await document_store.write_documents_async(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert await document_store.count_documents_async() == 3


class WriteDocumentsAsyncTest(AssertDocumentsEqualMixin):
    """
    Utility class to test a Document Store `write_documents_async` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    The Document Store `filter_documents_async` method must be at least partly implemented to return all stored
    Documents for these tests to work correctly.
    Example usage:

    ```python
    class MyDocumentStoreTest(WriteDocumentsAsyncTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store: AsyncDocumentStore):
        """
        Test write_documents_async() default behaviour.
        """
        msg = (
            "Default write_documents_async() behaviour depends on the Document Store implementation, "
            "as we don't enforce a default behaviour when no policy is set. "
            "Override this test in your custom test class."
        )
        raise NotImplementedError(msg)

    @pytest.mark.asyncio
    async def test_write_documents_duplicate_fail_async(self, document_store: AsyncDocumentStore):
        """Test write_documents_async() fails when writing documents with same id and `DuplicatePolicy.FAIL`."""
        doc = Document(content="test doc")
        assert await document_store.write_documents_async([doc], policy=DuplicatePolicy.FAIL) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(documents=[doc], policy=DuplicatePolicy.FAIL)
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [doc])

    @staticmethod
    @pytest.mark.asyncio
    async def test_write_documents_duplicate_skip_async(document_store: AsyncDocumentStore):
        """Test write_documents_async() skips writing when using DuplicatePolicy.SKIP."""
        doc = Document(content="test doc")
        assert await document_store.write_documents_async([doc], policy=DuplicatePolicy.SKIP) == 1
        assert await document_store.write_documents_async(documents=[doc], policy=DuplicatePolicy.SKIP) == 0

    @pytest.mark.asyncio
    async def test_write_documents_duplicate_overwrite_async(self, document_store: AsyncDocumentStore):
        """Test write_documents_async() overwrites when using DuplicatePolicy.OVERWRITE."""
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert await document_store.write_documents_async([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [doc2])
        assert await document_store.write_documents_async(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [doc1])

    @staticmethod
    @pytest.mark.asyncio
    async def test_write_documents_invalid_input_async(document_store: AsyncDocumentStore):
        """Test write_documents_async() fails when providing unexpected input."""
        with pytest.raises(ValueError):
            await document_store.write_documents_async(["not a document for sure"])  # type: ignore
        with pytest.raises(ValueError):
            await document_store.write_documents_async("not a list actually")  # type: ignore


class DeleteDocumentsAsyncTest:
    """
    Utility class to test a Document Store `delete_documents_async` method.

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    The Document Store `write_documents_async` and `count_documents_async` methods must be implemented for these tests
    to work correctly.
    Example usage:

    ```python
    class MyDocumentStoreTest(DeleteDocumentsAsyncTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_documents_async(document_store: AsyncDocumentStore):
        """Test delete_documents_async() normal behaviour."""
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_documents_empty_document_store_async(document_store: AsyncDocumentStore):
        """Test delete_documents_async() doesn't fail when called using an empty Document Store."""
        await document_store.delete_documents_async(["non_existing_id"])

    @staticmethod
    @pytest.mark.asyncio
    async def test_delete_documents_non_existing_document_async(document_store: AsyncDocumentStore):
        """Test delete_documents_async() doesn't delete any Document when called with non-existing id."""
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert await document_store.count_documents_async() == 1

        await document_store.delete_documents_async(["non_existing_id"])

        # No Document has been deleted
        assert await document_store.count_documents_async() == 1
