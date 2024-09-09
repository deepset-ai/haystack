# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import deserialize_document_store_in_init_params_inplace

logger = logging.getLogger(__name__)


@component
class DocumentWriter:
    """
    Writes documents to a DocumentStore.

    ### Usage example
    ```python
    from haystack import Document
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    docs = [
        Document(content="Python is a popular programming language"),
    ]
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs)
    ```
    """

    def __init__(self, document_store: DocumentStore, policy: DuplicatePolicy = DuplicatePolicy.NONE):
        """
        Create a DocumentWriter component.

        :param document_store:
            The instance of the document store where you want to store your documents.
        :param policy:
            The policy to apply when a Document with the same ID already exists in the DocumentStore.
            - `DuplicatePolicy.NONE`: Default policy, relies on the DocumentStore settings.
            - `DuplicatePolicy.SKIP`: Skips documents with the same ID and doesn't write them to the DocumentStore.
            - `DuplicatePolicy.OVERWRITE`: Overwrites documents with the same ID.
            - `DuplicatePolicy.FAIL`: Raises an error if a Document with the same ID is already in the DocumentStore.
        """
        self.document_store = document_store
        self.policy = policy

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
        return default_to_dict(self, document_store=self.document_store.to_dict(), policy=self.policy.name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentWriter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.

        :raises DeserializationError:
            If the document store is not properly specified in the serialization data or its type cannot be imported.
        """
        # deserialize the document store
        deserialize_document_store_in_init_params_inplace(data)

        data["init_parameters"]["policy"] = DuplicatePolicy[data["init_parameters"]["policy"]]

        return default_from_dict(cls, data)

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document], policy: Optional[DuplicatePolicy] = None):
        """
        Run the DocumentWriter on the given input data.

        :param documents:
            A list of documents to write to the document store.
        :param policy:
            The policy to use when encountering duplicate documents.
        :returns:
            Number of documents written to the document store.

        :raises ValueError:
            If the specified document store is not found.
        """
        if policy is None:
            policy = self.policy

        documents_written = self.document_store.write_documents(documents=documents, policy=policy)
        return {"documents_written": documents_written}
