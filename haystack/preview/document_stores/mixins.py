from typing import List, Type, Optional

from haystack.preview.document_stores.protocols import Store


class StoreAwareMixin:
    """
    Adds the capability of a component to use a single document store from the `self.store` property.
    """

    _store: Store
    _supported_stores: List[Type[Store]]

    @property
    def store(self) -> Optional[Store]:
        return self._store

    @store.setter
    def store(self, store: Store):
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        if self._supported_stores and any(not isinstance(storetype, store) for storetype in self._supported_stores):
            raise ValueError(
                f"Store type {type(store)} is not compatible with this component. "
                f"Compatible store types: {self._supported_stores}"
            )
        self._store = store
