# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore
from haystack.utils import deserialize_document_store_in_init_params_inplace

logger = logging.getLogger(__name__)


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

    def __init__(self, document_store: DocumentStore, cache_field: str):
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, document_store=self.document_store.to_dict(), cache_field=self.cache_field)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheChecker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        # deserialize the document store
        deserialize_document_store_in_init_params_inplace(data)

        return default_from_dict(cls, data)

    @component.output_types(hits=List[Document], misses=List)
    def run(self, items: List[Any]):
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
            if found:
                found_documents.extend(found)
            else:
                misses.append(item)
        return {"hits": found_documents, "misses": misses}
