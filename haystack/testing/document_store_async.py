# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.lazy_imports import LazyImport
from haystack.testing.document_store import AssertDocumentsEqualMixin

with LazyImport("Run 'pip install pytest'") as pytest_import:
    import pytest


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
