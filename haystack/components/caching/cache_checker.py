# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore
from haystack.utils.filters import _parse_iso_date


@component
class CacheChecker:
    """
    Checks for the presence of documents in a Document Store based on a specified field in each document's metadata.

    If matching documents are found, they are returned as "hits". If not found in the cache, the items
    are returned as "misses".

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
        *,
        ttl: float | timedelta | None = None,
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
            Time-to-live for cache entries, as a `timedelta` or a number of seconds.
            If set, a matching document only counts as a hit if the timestamp in `time_field` is younger than `ttl`.
            Documents with a missing or expired timestamp are returned as misses.
            If `None`, cache entries never expire.
        :param time_field:
            Name of the document's metadata field holding the time the document was cached,
            as a `datetime` or an ISO 8601 string. Only used when `ttl` is set.
        """
        if ttl is not None:
            if isinstance(ttl, bool):
                raise ValueError("ttl must be None, a positive number of seconds, or a positive timedelta.")

            if isinstance(ttl, (int, float)):
                if not math.isfinite(ttl) or ttl <= 0:
                    raise ValueError("ttl must be a positive number of seconds.")

                ttl = timedelta(seconds=ttl)

            elif isinstance(ttl, timedelta):
                if ttl <= timedelta(0):
                    raise ValueError("ttl must be a positive timedelta.")

            else:
                raise ValueError("ttl must be None, a positive number of seconds, or a positive timedelta.")
        self.document_store = document_store
        self.cache_field = cache_field
        self.ttl = ttl
        self.time_field = time_field.removeprefix("meta.")

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store,
            cache_field=self.cache_field,
            ttl=self.ttl.total_seconds() if self.ttl is not None else None,
            time_field=self.time_field,
        )

    def _is_fresh(self, document: Document) -> bool:
        """
        Checks whether a cached document is still within its TTL.

        Always returns True when no `ttl` is configured, to preserve the original non-expiring behavior.

        :param document:
            The candidate cache-hit document.
        :returns:
            True if the document should count as a cache hit, False if it should be treated as expired.
        """
        if self.ttl is None:
            return True

        cached_at = document.meta.get(self.time_field)
        if cached_at is None:
            # ttl is enabled but the document was never stamped with a cache time: treat as stale
            return False

        if isinstance(cached_at, str):
            cached_at = _parse_iso_date(cached_at)

        if not isinstance(cached_at, datetime):
            return False

        if cached_at.tzinfo is None:
            cached_at = cached_at.astimezone()

        age = datetime.now(UTC) - cached_at
        return timedelta(0) <= age < self.ttl

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
            fresh = [doc for doc in found if self._is_fresh(doc)]
            if fresh:
                found_documents.extend(fresh)
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
            fresh = [doc for doc in found if self._is_fresh(doc)]
            if fresh:
                found_documents.extend(fresh)
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
