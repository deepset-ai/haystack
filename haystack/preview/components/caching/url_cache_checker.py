from typing import List, Dict, Any

from haystack.preview import component, Document, default_from_dict, default_to_dict, DeserializationError
from haystack.preview.document_stores import DocumentStore, DuplicatePolicy, document_store


@component
class UrlCacheChecker:
    """
    A component that check if a document coming from a given URL is already present in the store.

    Can be used to implement a caching functionality with a Document Store in web retrieval pipelines.
    """

    def __init__(self, document_store: DocumentStore, url_field: str = "url"):
        """
        Create a UrlCacheChecker component.

        :param policy: The policy to use when encountering duplicate documents (default is DuplicatePolicy.FAIL).
        """
        self.document_store = document_store
        self.url_field = url_field

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, document_store=self.document_store.to_dict(), url_field=self.url_field)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UrlCacheChecker":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")
        if init_params["document_store"]["type"] not in document_store.registry:
            raise DeserializationError(f"DocumentStore of type '{init_params['document_store']['type']}' not found.")
        docstore_class = document_store.registry[init_params["document_store"]["type"]]
        docstore = docstore_class.from_dict(init_params["document_store"])

        data["init_parameters"]["document_store"] = docstore
        return default_from_dict(cls, data)

    @component.output_types(found=List[Document], missing=List[str])
    def run(self, urls: List[str]):
        """
        Checks if any document coming from the given URL is already present in the store and if so, returns it.

        :param urls: All the URLs the documents may be coming from to hit this cache.
        """
        found_documents = []
        missing_urls = []

        for url in urls:
            filters = {self.url_field: url}
            found = self.document_store.filter_documents(filters=filters)
            if found:
                found_documents.extend(found)
            else:
                missing_urls.append(url)
        return {"found": found_documents, "missing": missing_urls}
