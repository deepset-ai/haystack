# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Protocol

from haystack.dataclasses import Document
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.lazy_imports import LazyImport

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
