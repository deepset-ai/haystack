from typing import Protocol, Dict, Optional

from haystack.preview.document_stores.protocols import Store


class StoreComponent(Protocol):
    """
    Represents a component that needs a single store to run.
    """

    _store: Store

    @property
    def store(self) -> Optional[Store]:
        ...

    @store.setter
    def store(self, store: Store):
        ...


class StoreMixin(StoreComponent):
    """
    Adds the capability of a component to use a single document store from the `self.store` property.
    """

    @property
    def store(self) -> Optional[Store]:
        return getattr(self, "_store", None)

    @store.setter
    def store(self, store: Store):
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        self._store = store


class MultiStoreComponent:
    """
    Represents a component that needs more than a single store to run.
    """

    @property
    def stores(self) -> Optional[Dict[str, Store]]:
        ...

    @stores.setter
    def stores(self, stores: Dict[str, Store]):
        ...


class MultiStoreMixin(MultiStoreComponent):
    """
    Adds the capability of a component to use several document stores from the `self.stores` property.
    """

    _stores: Dict[str, Store]

    @property
    def stores(self) -> Optional[Dict[str, Store]]:
        return getattr(self, "_stores", None)

    @stores.setter
    def stores(self, stores: Dict[str, Store]):
        if stores is None:
            raise ValueError("The stores dictionary can't be None.")
        self._stores = stores
