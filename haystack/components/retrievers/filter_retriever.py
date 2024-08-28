# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore
from haystack.utils import deserialize_document_store_in_init_params_inplace

logger = logging.getLogger(__name__)


@component
class FilterRetriever:
    """
    Retrieves documents that match the provided filters.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.retrievers import FilterRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    docs = [
        Document(content="Python is a popular programming language", meta={"lang": "en"}),
        Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
    ]

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    retriever = FilterRetriever(doc_store, filters={"field": "lang", "operator": "==", "value": "en"})

    # if passed in the run method, filters override those provided at initialization
    result = retriever.run(filters={"field": "lang", "operator": "==", "value": "de"})

    print(result["documents"])
    ```
    """

    def __init__(self, document_store: DocumentStore, filters: Optional[Dict[str, Any]] = None):
        """
        Create the FilterRetriever component.

        :param document_store:
            An instance of a Document Store to use with the Retriever.
        :param filters:
            A dictionary with filters to narrow down the search space.
        """
        self.document_store = document_store
        self.filters = filters

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"document_store": type(self.document_store).__name__}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(self, document_store=docstore, filters=self.filters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        # deserialize the document store
        deserialize_document_store_in_init_params_inplace(data)

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, filters: Optional[Dict[str, Any]] = None):
        """
        Run the FilterRetriever on the given input data.

        :param filters:
            A dictionary with filters to narrow down the search space.
            If not specified, the FilterRetriever uses the values provided at initialization.
        :returns:
            A list of retrieved documents.
        """
        return {"documents": self.document_store.filter_documents(filters=filters or self.filters)}
