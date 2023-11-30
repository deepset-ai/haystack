from pydoc import doc
from typing import List, Optional, Dict, Any

from haystack import component, Document, default_from_dict, default_to_dict, DeserializationError
from haystack.document_stores import DocumentStore, DuplicatePolicy, document_store
import importlib

import logging

logger = logging.getLogger(__name__)


@component
class DocumentWriter:
    """
    A component for writing documents to a DocumentStore.
    """

    def __init__(self, document_store: DocumentStore, policy: DuplicatePolicy = DuplicatePolicy.FAIL):
        """
        Create a DocumentWriter component.

        :param policy: The policy to use when encountering duplicate documents (default is DuplicatePolicy.FAIL).
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
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, document_store=self.document_store.to_dict(), policy=self.policy.name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentWriter":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")

        try:
            # Import the module first...
            module, type_ = init_params["document_store"]["type"].rsplit(".", 1)
            logger.debug("Trying to import %s", module)
            module = importlib.import_module(module)
            # ...then try again
            # if init_params["document_store"]["type"] not in component.registry:
            #     raise DeserializationError(
            #         f"Successfully imported module {module} but can't find it in the component registry."
            #         "This is unexpected and most likely a bug."
            #     )
        except (ImportError, DeserializationError) as e:
            raise DeserializationError(f"Component {module} not imported.") from e

        docstore_class = getattr(module, type_)
        docstore = docstore_class.from_dict(init_params["document_store"])

        data["init_parameters"]["document_store"] = docstore
        data["init_parameters"]["policy"] = DuplicatePolicy[data["init_parameters"]["policy"]]
        return default_from_dict(cls, data)

    @component.output_types(documents_written=int)
    def run(self, documents: List[Document], policy: Optional[DuplicatePolicy] = None):
        """
        Run DocumentWriter on the given input data.

        :param documents: A list of documents to write to the store.
        :param policy: The policy to use when encountering duplicate documents.
        :return: Number of documents written

        :raises ValueError: If the specified document store is not found.
        """
        if policy is None:
            policy = self.policy

        documents_written = self.document_store.write_documents(documents=documents, policy=policy)
        return {"documents_written": documents_written}
