from typing import List, Dict, Any, Optional, Callable

from pathlib import Path

from canals.pipeline import (
    Pipeline as CanalsPipeline,
    PipelineError,
    load_pipelines as load_canals_pipelines,
    save_pipelines as save_canals_pipelines,
)

from haystack.preview.document_stores.protocols import DocumentStore
from haystack.preview.document_stores.mixins import DocumentStoreAwareMixin


class NotADocumentStoreError(PipelineError):
    pass


class NoSuchDocumentStoreError(PipelineError):
    pass


class Pipeline(CanalsPipeline):
    """
    Haystack Pipeline is a thin wrapper over Canals' Pipelines to add support for DocumentStores.
    """

    def __init__(self):
        super().__init__()
        self._document_stores: Dict[str, DocumentStore] = {}

    def add_document_store(self, name: str, document_store: DocumentStore) -> None:
        """
        Make a DocumentStore available to all nodes of this pipeline.

        :param name: the name of the DocumentStore.
        :param document_store: the DocumentStore object.
        :returns: None
        """
        if not getattr(document_store, "__haystack_document_store__", False):
            raise NotADocumentStoreError(
                f"'{type(document_store).__name__}' is not decorated with @document_store, "
                "so it can't be added to the pipeline with Pipeline.add_document_store()."
            )
        self._document_stores[name] = document_store

    def list_document_stores(self) -> List[str]:
        """
        Returns a dictionary with all the DocumentStores that are attached to this Pipeline.

        :returns: a dictionary with all the DocumentStores attached to this Pipeline.
        """
        return list(self._document_stores.keys())

    def get_document_store(self, name: str) -> DocumentStore:
        """
        Returns the DocumentStore associated with the given name.

        :param name: the name of the DocumentStore
        :returns: the DocumentStore
        """
        try:
            return self._document_stores[name]
        except KeyError as e:
            raise NoSuchDocumentStoreError(f"No DocumentStore named '{name}' was added to this pipeline.") from e

    def add_component(self, name: str, instance: Any, document_store: Optional[str] = None) -> None:
        """
        Make this component available to the pipeline. Components are not connected to anything by default:
        use `Pipeline.connect()` to connect components together.

        Component names must be unique, but component instances can be reused if needed.

        If `document_store` is not None, the pipeline will also connect this component to the requested DocumentStore.
        Note that only components that inherit from DocumentStoreAwareMixin can be connected to DocumentStores.

        :param name: the name of the component.
        :param instance: the component instance.
        :param document_store: the DocumentStore this component needs access to, if any.
        :raises ValueError: if:
            - a component with the same name already exists
            - a component requiring a DocumentStore didn't receive it
            - a component that didn't expect a DocumentStore received it
        :raises PipelineValidationError: if the given instance is not a component
        :raises NoSuchDocumentStoreError: if the given DocumentStore name is not known to the pipeline
        """
        if isinstance(instance, DocumentStoreAwareMixin):
            if not document_store:
                raise ValueError(f"Component '{name}' needs a DocumentStore.")

            if document_store not in self._document_stores:
                raise NoSuchDocumentStoreError(
                    f"DocumentStore named '{document_store}' not found. "
                    f"Add it with 'pipeline.add_document_store('{document_store}', <the DocumentStore instance>)'."
                )

            if instance.document_store:
                raise ValueError("Reusing components with DocumentStores is not supported. Create a separate instance.")

            instance.document_store = self._document_stores[document_store]
            instance._document_store_name = document_store

        elif document_store:
            raise ValueError(f"Component '{name}' doesn't support DocumentStores.")

        super().add_component(name, instance)


def load_pipelines(path: Path, _reader: Optional[Callable[..., Any]] = None):
    return load_canals_pipelines(path=path, _reader=_reader)


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer: Optional[Callable[..., Any]] = None):
    save_canals_pipelines(pipelines=pipelines, path=path, _writer=_writer)
