from typing import List, Optional, Type


from haystack.preview.document_stores.protocols import DocumentStore


class DocumentStoreAwareMixin:
    """
    Adds the capability of a component to use a single DocumentStore from the `self.document_store` property.

    To use this mixin you must specify which DocumentStores to support by setting a value to `supported_stores`.
    To support any DocumentStore, set it to `[DocumentStore]`.
    """

    _document_store: Optional[DocumentStore] = None
    # This is necessary to ease serialisation when converting a Component that uses
    # a DocumentStore into a dictionary.
    # This is only set when calling `Pipeline.add_component()`.
    _document_store_name: str = ""
    supported_document_stores: List[Type[DocumentStore]]  # type: ignore # (see https://github.com/python/mypy/issues/4717)

    @property
    def document_store(self) -> Optional[DocumentStore]:
        return self._document_store

    @document_store.setter
    def document_store(self, document_store: DocumentStore):
        if not getattr(document_store, "__haystack_document_store__", False):
            raise ValueError(f"'{type(document_store).__name__}' is not decorate with @document_store.")
        if not self._is_supported(document_store):
            raise ValueError(
                f"DocumentStore type '{type(document_store).__name__}' is not compatible with this component. "
                f"Compatible DocumentStore types: {[type_.__name__ for type_ in type(self).supported_document_stores]}"
            )
        self._document_store = document_store

    def _is_supported(self, document_store: DocumentStore):
        if DocumentStore in self.supported_document_stores:
            return True
        return any(isinstance(document_store, type_) for type_ in self.supported_document_stores)
