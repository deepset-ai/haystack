from typing import List, Dict, Any, Optional, Callable

from pathlib import Path

from canals.pipeline import (
    Pipeline as CanalsPipeline,
    PipelineError,
    load_pipelines as load_canals_pipelines,
    save_pipelines as save_canals_pipelines,
)

from haystack.preview.document_stores.protocols import Store
from haystack.preview.document_stores.mixins import StoreAwareMixin


class NotAStoreError(PipelineError):
    pass


class NoSuchStoreError(PipelineError):
    pass


class Pipeline(CanalsPipeline):
    """
    Haystack Pipeline is a thin wrapper over Canals' Pipelines to add support for Stores.
    """

    def __init__(self):
        super().__init__()
        self._stores: Dict[str, Store] = {}

    def add_store(self, name: str, store: Store) -> None:
        """
        Make a store available to all nodes of this pipeline.

        :param name: the name of the store.
        :param store: the store object.
        :returns: None
        """
        if not isinstance(store, Store):
            raise NotAStoreError(
                f"This object ({store}) does not respect the Store Protocol, "
                "so it can't be added to the pipeline with Pipeline.add_store()."
            )
        self._stores[name] = store

    def list_stores(self) -> List[str]:
        """
        Returns a dictionary with all the stores that are attached to this Pipeline.

        :returns: a dictionary with all the stores attached to this Pipeline.
        """
        return list(self._stores.keys())

    def get_store(self, name: str) -> Store:
        """
        Returns the store associated with the given name.

        :param name: the name of the store
        :returns: the store
        """
        try:
            return self._stores[name]
        except KeyError as e:
            raise NoSuchStoreError(f"No store named '{name}' is connected to this pipeline.") from e

    def add_component(self, name: str, instance: Any, store: Optional[str] = None) -> None:
        """
        Make this component available to the pipeline. Components are not connected to anything by default:
        use `Pipeline.connect()` to connect components together.

        Component names must be unique, but component instances can be reused if needed.

        If `store` has a value, the pipeline will also connect this component to the requested document store.
        Note that only components that inherit from StoreAwareMixin can be connected to stores.

        :param name: the name of the component.
        :param instance: the component instance.
        :param store: the store this component needs access to, if any.
        :raises ValueError: if:
            - a component with the same name already exists
            - a component requiring a store didn't receive it
            - a component that didn't expect a store received it
        :raises PipelineValidationError: if the given instance is not a component
        :raises NoSuchStoreError: if the given store name is not known to the pipeline
        """
        if isinstance(instance, StoreAwareMixin):
            if not store:
                raise ValueError(f"Component '{name}' needs a store.")

            if store not in self._stores:
                raise NoSuchStoreError(
                    f"Store named '{store}' not found. "
                    f"Add it with 'pipeline.add_store('{store}', <the docstore instance>)'."
                )

            if instance.store:
                raise ValueError("Reusing components with stores is not supported (yet). Create a separate instance.")

            instance.store = self._stores[store]

        elif store:
            raise ValueError(f"Component '{name}' doesn't support stores.")

        super().add_component(name, instance)


def load_pipelines(path: Path, _reader: Optional[Callable[..., Any]] = None):
    return load_canals_pipelines(path=path, _reader=_reader)


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer: Optional[Callable[..., Any]] = None):
    save_canals_pipelines(pipelines=pipelines, path=path, _writer=_writer)
