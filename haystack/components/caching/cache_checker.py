# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore


@component
class CacheChecker:
    """
    Checks for the presence of documents in a Document Store based on a specified field in each document's metadata.

    If matching documents are found, they are returned as "hits". If not found in the cache, the items
    are returned as "misses".

    If `ttl` is provided, matching documents are considered cache hits only if their
    timestamp is within the specified TTL. The timestamp is read from the `time_field`
    metadata field, which defaults to `"cached_at"`. If `ttl` is `None`, documents are
    considered cache hits based only on the presence of a matching `cache_field`.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.caching.cache_checker import CacheChecker

    docstore = InMemoryDocumentStore()
    documents = [
        Document(content="doc1", meta={"url": "https://example.com/1"}),
        Document(content="doc2", meta={"url": "https://example.com/2"}),
        Document(content="doc3", meta={"url": "https://example.com/1"}),
        Document(content="doc4", meta={"url": "https://example.com/2"}),
    ]
    docstore.write_documents(documents)
    checker = CacheChecker(docstore, cache_field="url")
    results = checker.run(items=["https://example.com/1", "https://example.com/5"])
    assert results == {"hits": [documents[0], documents[2]], "misses": ["https://example.com/5"]}
    ```
    """

    def __init__(
        self,
        document_store: DocumentStore,
        cache_field: str,
        ttl: timedelta | None = None,
        time_field: str = "cached_at",
    ) -> None:
        """
        Creates a CacheChecker component.

        :param document_store:
            Document Store to check for the presence of specific documents.
        :param cache_field:
            Name of the document's metadata field
            to check for cache hits.
        :param ttl:
            Maximum age of a cached document before it is considered expired. If `None`,
            matching documents are considered cache hits regardless of their age.
        :param time_field:
            Name of the document's metadata field containing the cache timestamp.
            Defaults to `"cached_at"`.
        """
        self.document_store = document_store
        self.cache_field = cache_field
        self.ttl = ttl
        self.time_field = time_field

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        init_parameters = {"document_store": self.document_store, "cache_field": self.cache_field}

        if self.ttl is not None:
            init_parameters["ttl"] = self.ttl

        if self.time_field != "cached_at":
            init_parameters["time_field"] = self.time_field

        return default_to_dict(self, **init_parameters)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheChecker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def _filter_valid_documents(self, documents: list[Document]) -> list[Document]:
        if self.ttl is None:
            return documents

        now = datetime.now()

        return [
            document
            for document in documents
            if self.time_field in document.meta and now - document.meta[self.time_field] < self.ttl
        ]

    @component.output_types(hits=list[Document], misses=list)
    def run(self, items: list[Any]) -> dict[str, Any]:
        """
        Checks if any document associated with the specified cache field is already present in the store.

        :param items:
            Values to be checked against the cache field.
        :return:
            A dictionary with two keys:
            - `hits` - Documents that matched with at least one of the items.
            - `misses` - Items that were not present in any documents.
        """
        found_documents = []
        misses = []

        for item in items:
            filters = {"field": self.cache_field, "operator": "==", "value": item}
            found = self.document_store.filter_documents(filters=filters)
            valid_documents = self._filter_valid_documents(found)
            if valid_documents:
                found_documents.extend(valid_documents)
            else:
                misses.append(item)
        return {"hits": found_documents, "misses": misses}

    @component.output_types(hits=list[Document], misses=list)
    async def run_async(self, items: list[Any]) -> dict[str, Any]:
        """
        Asynchronously checks if any document associated with the specified cache field is already present in the store.

        :param items:
            Values to be checked against the cache field.
        :return:
            A dictionary with two keys:
            - `hits` - Documents that matched with at least one of the items.
            - `misses` - Items that were not present in any documents.
        """
        found_documents = []
        misses = []

        if not hasattr(self.document_store, "filter_documents_async"):
            raise TypeError(f"Document store {type(self.document_store).__name__} does not provide async support.")

        for item in items:
            filters = {"field": self.cache_field, "operator": "==", "value": item}
            found = await self.document_store.filter_documents_async(filters=filters)
            valid_documents = self._filter_valid_documents(found)
            if valid_documents:
                found_documents.extend(valid_documents)
            else:
                misses.append(item)
        return {"hits": found_documents, "misses": misses}

    def close(self) -> None:
        """
        Release the synchronous resources of the underlying Document Store.
        """
        if hasattr(self.document_store, "close"):
            self.document_store.close()

    async def close_async(self) -> None:
        """
        Release the asynchronous resources of the underlying Document Store.
        """
        if hasattr(self.document_store, "close_async"):
            await self.document_store.close_async()
