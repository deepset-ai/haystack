from typing import List, Dict, Any

import importlib

import logging

from haystack import component, Document, default_from_dict, default_to_dict, DeserializationError
from haystack.document_stores.types import DocumentStore


logger = logging.getLogger(__name__)


@component
class CacheChecker:
    """
    CacheChecker is a component that checks for the presence of documents in a Document Store based on a specified
    cache field.
    """

    def __init__(self, document_store: DocumentStore, cache_field: str):
        """
        Create a UrlCacheChecker component.
        """
        self.document_store = document_store
        self.cache_field = cache_field

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, document_store=self.document_store.to_dict(), cache_field=self.cache_field)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheChecker":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")

        try:
            module_name, type_ = init_params["document_store"]["type"].rsplit(".", 1)
            logger.debug("Trying to import %s", module_name)
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
        Checks if any document associated with the specified field is already present in the store. If matching documents
        are found, they are returned as hits. If not, the items are returned as misses, indicating they are not in the cache.

        :param items: A list of values associated with the cache_field to be checked against the cache.
        :return: A dictionary with two keys: "hits" and "misses". The values are lists of documents that were found in
        the cache and items that were not, respectively.
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
