from typing import List, Type, Optional, Any

from haystack.preview.document_stores.protocols import Store


class StoreAwareMixin:
    """
    Adds the capability of a component to use a single document store from the `self.store` property.
    """

    _store: Store
    _supported_stores: List[Store]

    @property
    def store(self) -> Optional[Store]:
        return self._store

    @store.setter
    def store(self, store: Store):
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        if hasattr(self, "_supported_stores") and any(not isinstance(store, type_) for type_ in self._supported_stores):
            raise ValueError(
                f"Store type '{type(store).__name__}' is not compatible with this component. "
                f"Compatible store types: {[type_.__name__ for type_ in self._supported_stores]}"
            )
        self._store = store
