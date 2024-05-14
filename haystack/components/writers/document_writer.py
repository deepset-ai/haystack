# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

logger = logging.getLogger(__name__)


@component
class DocumentWriter:
    """
    Writes documents to a DocumentStore.

    Usage example:
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
            The DocumentStore where the documents are to be written.
        :param policy:
            The policy to apply when a Document with the same id already exists in the DocumentStore.
            - `DuplicatePolicy.NONE`: Default policy, behaviour depends on the Document Store.
            - `DuplicatePolicy.SKIP`: If a Document with the same id already exists, it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
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
        data["init_parameters"]["policy"] = DuplicatePolicy[data["init_parameters"]["policy"]]
        return default_from_dict(cls, data)

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document], policy: Optional[DuplicatePolicy] = None):
        """
        Run the DocumentWriter on the given input data.

        :param documents:
            A list of documents to write to the store.
        :param policy:
            The policy to use when encountering duplicate documents.
        :returns:
            Number of documents written

        :raises ValueError:
            If the specified document store is not found.
        """
        if policy is None:
            policy = self.policy

        documents_written = self.document_store.write_documents(documents=documents, policy=policy)
        return {"documents_written": documents_written}
