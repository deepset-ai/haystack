from typing import Dict, Any, Type
import logging

from haystack.preview.document_stores.protocols import DocumentStore
from haystack.preview.document_stores.errors import DocumentStoreDeserializationError

logger = logging.getLogger(__name__)


class _DocumentStore:
    """
    Marks a class as an Haystack _DocumentStore.
    All classes decorated with @document_store will be registered here and can be used in Haystack Pipelines.
    """

    def __init__(self):
        self.registry = {}

    def _decorate(self, cls):
        cls.__haystack_document_store__ = True

        if cls.__name__ in self.registry:
            logger.error(
                "DocumentStore %s is already registered. Previous imported from '%s', new imported from '%s'",
                cls.__name__,
                self.registry[cls.__name__],
                cls,
            )

        self.registry[cls.__name__] = cls
        logger.debug("Registered DocumentStore %s", cls)

        return cls

    def __call__(self, cls=None):
        if cls:
            return self._decorate(cls)

        return self._decorate


document_store = _DocumentStore()


def default_document_store_to_dict(store_: DocumentStore) -> Dict[str, Any]:
    """
    Default DocumentStore serializer.
    Serializes a DocumentStore to a dictionary.
    """
    return {
        "hash": id(store_),
        "type": store_.__class__.__name__,
        "init_parameters": getattr(store_, "init_parameters", {}),
    }


def default_document_store_from_dict(cls: Type[DocumentStore], data: Dict[str, Any]) -> DocumentStore:
    """
    Default DocumentStore deserializer.
    The "type" field in `data` must match the class that is being deserialized into.
    """
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise DocumentStoreDeserializationError("Missing 'type' in DocumentStore serialization data")
    if data["type"] != cls.__name__:
        raise DocumentStoreDeserializationError(
            f"DocumentStore '{data['type']}' can't be deserialized as '{cls.__name__}'"
        )
    return cls(**init_params)
