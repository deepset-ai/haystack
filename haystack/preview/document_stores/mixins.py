from typing import Protocol, Dict

from haystack.preview.document_stores.protocols import Store


class StoreMixin(Protocol):
    """
    Adds the capability of a component to use a single document store from the `self.store` property.
    """

    _store: Store

    @property
    def store(self) -> Store:
        return getattr(self, "_store", None)

    @store.setter
    def store(self, store: Store):
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        self._store = store


class MultiStoreMixin(Protocol):
    """
    Adds the capability of a component to use several document stores from the `self.stores` property.
    """

    _stores: Dict[str, Store]

    @property
    def stores(self) -> Dict[str, Store]:
        return getattr(self, "_stores", None)

    @stores.setter
    def stores(self, stores: Dict[str, Store]):
        if stores is None:
            raise ValueError("The stores dictionary can't be None.")
        self._stores = stores
