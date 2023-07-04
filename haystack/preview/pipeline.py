from typing import List, Dict, Any, Optional, Callable

from pathlib import Path

from canals.pipeline import (
    Pipeline as CanalsPipeline,
    PipelineError,
    load_pipelines as load_canals_pipelines,
    save_pipelines as save_canals_pipelines,
)

from haystack.preview.document_stores.protocols import Store


class NoSuchStoreError(PipelineError):
    pass


class Pipeline(CanalsPipeline):
    """
    Haystack Pipeline is a thin wrapper over Canals' Pipelines to add support for Stores.
    """

    def __init__(self):
        super().__init__()
        self.stores: Dict[str, Store] = {}

    def add_store(self, name: str, store: Store) -> None:
        """
        Make a store available to all nodes of this pipeline.

        :param name: the name of the store.
        :param store: the store object.
        :returns: None
        """
        self.stores[name] = store

    def list_stores(self) -> List[str]:
        """
        Returns a dictionary with all the stores that are attached to this Pipeline.

        :returns: a dictionary with all the stores attached to this Pipeline.
        """
        return list(self.stores.keys())

    def get_store(self, name: str) -> Store:
        """
        Returns the store associated with the given name.

        :param name: the name of the store
        :returns: the store
        """
        try:
            return self.stores[name]
        except KeyError as e:
            raise NoSuchStoreError(f"No store named '{name}' is connected to this pipeline.") from e

    def add_component(self, name: str, instance: Any, stores: Optional[List[str]] = None) -> None:
        """
        Make this component available to the pipeline. Components are not connected to anything by default:
        use `Pipeline.connect()` to connect components together.

        Component names must be unique, but component instances can be reused if needed.

        If `stores` has a value, the pipeline will also connect this component to the requested document store(s).
        Note that only components that respect the StoreMixin or StoresMixin protocols can be connected to stores.

        :param name: the name of the component.
        :param instance: the component instance.
        :param stores: the stores this component needs access to, if any.
        :raises ValueError: if a component with the same name already exists
        :raises PipelineValidationError: if the given instance is not a component
        """
        super().add_component(name, instance)

        stores = stores or []
        for store in stores:
            if store not in self.stores:
                raise NoSuchStoreError(
                    f"Store named '{store}' not found. "
                    f"Add it with 'pipeline.add_store('{store}', <the docstore instance>)'."
                )

        # This component implements the MultiStoreComponent protocol
        if hasattr(instance, "stores"):
            instance.stores = {store: self.stores[store] for store in stores}

        # This component implements the StoreComponent protocol
        elif hasattr(instance, "store"):
            if len(stores) != 1:
                raise ValueError(f"Component '{name}' needs exactly one store.")
            instance.store = self.stores[stores[0]]


def load_pipelines(path: Path, _reader: Optional[Callable[..., Any]] = None):
    return load_canals_pipelines(path=path, _reader=_reader)


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer: Optional[Callable[..., Any]] = None):
    save_canals_pipelines(pipelines=pipelines, path=path, _writer=_writer)
