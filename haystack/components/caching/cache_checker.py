# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Dict, List

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore

logger = logging.getLogger(__name__)


@component
class CacheChecker:
    """
    Checks for the presence of documents in a Document Store based on a specified field in each document's metadata.

    If matching documents are found, they are returned as hits. If not, the items
    are returned as misses, indicating they are not in the cache.

    Usage example:
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
        Create a CacheChecker component.

        :param document_store:
            Document store to check.
        :param cache_field:
            Name of the Document metadata field
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
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")

        try:
            module_name, type_ = init_params["document_store"]["type"].rsplit(".", 1)
            logger.debug("Trying to import module '{module_name}'", module_name=module_name)
            module = importlib.import_module(module_name)
        except (ImportError, DeserializationError) as e:
            raise DeserializationError(
                f"DocumentStore of type '{init_params['document_store']['type']}' not correctly imported"
            ) from e

        docstore_class = getattr(module, type_)
        docstore = docstore_class.from_dict(init_params["document_store"])

        data["init_parameters"]["document_store"] = docstore
        return default_from_dict(cls, data)

    @component.output_types(hits=List[Document], misses=List)
    def run(self, items: List[Any]):
        """
        Checks if any document associated with the specified cache field is already present in the store.

        :param items:
            Values to be checked against the cache field.
        :return:
            A dictionary with two keys:
            - `hits` - Documents that matched with any of the items.
            - `misses` - Items that were not present in any documents.
        """
        found_documents = []
        misses = []

        for item in items:
            filters = {self.cache_field: item}
            found = self.document_store.filter_documents(filters=filters)
            if found:
                found_documents.extend(found)
            else:
                misses.append(item)
        return {"hits": found_documents, "misses": misses}
