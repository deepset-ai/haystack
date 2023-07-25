from typing import List, Optional, Type


from haystack.preview.document_stores.protocols import Store


class StoreAwareMixin:
    """
    Adds the capability of a component to use a single document store from the `self.store` property.

    To use this mixin you must specify which document stores to support by setting a value to `supported_stores`.
    To support any document store, set it to `[Store]`.
    """

    _store: Optional[Store] = None
    supported_stores: List[Type[Store]]  # type: ignore # (see https://github.com/python/mypy/issues/4717)

    @property
    def store(self) -> Optional[Store]:
        return self._store

    @store.setter
    def store(self, store: Store):
        if not isinstance(store, Store):
            raise ValueError("'store' does not respect the Store Protocol.")
        if not any(isinstance(store, type_) for type_ in type(self).supported_stores):
            raise ValueError(
                f"Store type '{type(store).__name__}' is not compatible with this component. "
                f"Compatible store types: {[type_.__name__ for type_ in type(self).supported_stores]}"
            )
        self._store = store
