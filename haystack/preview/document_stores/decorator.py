from typing import Dict, Any, Type
import logging

from haystack.preview.document_stores.protocols import Store
from haystack.preview.document_stores.errors import StoreDeserializationError

logger = logging.getLogger(__name__)


class _Store:
    """
    Marks a class as an Haystack Store.
    All classes decorated with @store will be registered here and can be used in Haystack Pipelines.
    """

    def __init__(self):
        self.registry = {}

    def _decorate(self, cls):
        cls.__haystack_store__ = True

        if cls.__name__ in self.registry:
            logger.error(
                "Store %s is already registered. Previous imported from '%s', new imported from '%s'",
                cls.__name__,
                self.registry[cls.__name__],
                cls,
            )

        self.registry[cls.__name__] = cls
        logger.debug("Registered Store %s", cls)

        cls.to_dict = _default_store_to_dict
        cls.from_dict = classmethod(_default_store_from_dict)

        return cls

    def __call__(self, cls=None):
        if cls:
            return self._decorate(cls)

        return self._decorate


store = _Store()


def _default_store_to_dict(store_: Store) -> Dict[str, Any]:
    """
    Default store serializer.
    Serializes a store to a dictionary.
    """
    return {
        "hash": id(store_),
        "type": store_.__class__.__name__,
        "init_parameters": getattr(store_, "init_parameters", {}),
    }


def _default_store_from_dict(cls: Type[Store], data: Dict[str, Any]) -> Store:
    """
    Default store deserializer.
    The "type" field in `data` must match the class that is being deserialized into.
    """
    init_params = data.get("init_parameters", {})
    if "type" not in data:
        raise StoreDeserializationError("Missing 'type' in store serialization data")
    if data["type"] != cls.__name__:
        raise StoreDeserializationError(f"Store '{data['type']}' can't be deserialized as '{cls.__name__}'")
    return cls(**init_params)
