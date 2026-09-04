# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore
from haystack.utils import document_matches_filter


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

    def __init__(self, document_store: DocumentStore, cache_field: str) -> None:
        """
        Creates a CacheChecker component.

        :param document_store:
            Document Store to check for the presence of specific documents.
        :param cache_field:
            Name of the document's metadata field
            to check for cache hits.
        """
        self.document_store = document_store
        self.cache_field = cache_field

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, document_store=self.document_store, cache_field=self.cache_field)

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
        if not items:
            return {"hits": [], "misses": []}

        candidates = self.document_store.filter_documents(
            filters={"field": self.cache_field, "operator": "in", "value": items}
        )
        return self._split_hits_and_misses(items, candidates)

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
        if not hasattr(self.document_store, "filter_documents_async"):
            raise TypeError(f"Document store {type(self.document_store).__name__} does not provide async support.")

        if not items:
            return {"hits": [], "misses": []}

        candidates = await self.document_store.filter_documents_async(
            filters={"field": self.cache_field, "operator": "in", "value": items}
        )
        return self._split_hits_and_misses(items, candidates)

    def _split_hits_and_misses(self, items: list[Any], candidates: list[Document]) -> dict[str, Any]:
        """
        Groups the documents returned by a single batched lookup back onto the items that matched them.

        The store is queried once with the `in` operator; this reproduces, in process, the per-item grouping
        that one `==` query per item used to produce. `document_matches_filter` is the same predicate the
        filtering machinery applies, so field resolution and value comparison stay identical. Stores that
        can compare datetimes strictly expose that choice as `strict_datetime_comparison`; it is read here
        so the grouping matches the query the store just answered.

        :param items:
            Values that were checked against the cache field, in the caller's order.
        :param candidates:
            Documents returned by the batched lookup.
        :returns:
            A dictionary with `hits` and `misses`.
        """
        strict_datetime_comparison = getattr(self.document_store, "strict_datetime_comparison", False)
        found_documents = []
        misses = []

        for item in items:
            condition = {"field": self.cache_field, "operator": "==", "value": item}
            found = [
                document
                for document in candidates
                if document_matches_filter(condition, document, strict_datetime_comparison=strict_datetime_comparison)
            ]
            if found:
                found_documents.extend(found)
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
