from typing import Dict, Optional

from haystack.preview.document_stores.protocols import Store


class StoreMixin:
    """
    Adds the capability of a component to use a single document store from the `self.store` property.
    """

    _store: Store

    @property
    def store(self) -> Optional[Store]:
        return self._store

    @store.setter
    def store(self, store: Store):
        if not store:
            raise ValueError("Can't set the value of the store to None.")
        self._store = store


class MultiStoreMixin:
    """
    Adds the capability of a component to use several document stores from the `self.stores` property.
    """

    _stores: Dict[str, Store]

    @property
    def stores(self) -> Optional[Dict[str, Store]]:
        return self._stores

    @stores.setter
    def stores(self, stores: Dict[str, Store]):
        if stores is None:
            raise ValueError("The stores dictionary can't be None.")
        self._stores = stores
