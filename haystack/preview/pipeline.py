from typing import List, Dict, Any, Optional, Callable

from pathlib import Path

from canals.component import ComponentInput
from canals.pipeline import (
    Pipeline as CanalsPipeline,
    PipelineError,
    load_pipelines as load_canals_pipelines,
    save_pipelines as save_canals_pipelines,
)
from canals.pipeline.sockets import find_input_sockets


class NoSuchStoreError(PipelineError):
    pass


class Pipeline(CanalsPipeline):
    """
    Haystack Pipeline is a thin wrapper over Canals' Pipelines to add support for Stores.
    """

    def __init__(self):
        super().__init__()
        self.stores = {}

    def add_store(self, name: str, store: object) -> None:
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

    def get_store(self, name: str) -> object:
        """
        Returns the store associated with the given name.

        :param name: the name of the store
        :returns: the store
        """
        try:
            return self.stores[name]
        except KeyError as e:
            raise NoSuchStoreError(f"No store named '{name}' is connected to this pipeline.") from e

    def run(self, data: Dict[str, ComponentInput], debug: bool = False):
        """
        Wrapper on top of Canals Pipeline.run(). Adds the `stores` parameter to all nodes.
        """
        # Get all nodes in this pipelines instance
        for node_name in self.graph.nodes:
            # Get node inputs
            node = self.graph.nodes[node_name]["instance"]
            input_params = find_input_sockets(node)

            # If the node needs a store, adds the list of stores to its default inputs
            if "stores" in input_params:
                if not hasattr(node, "defaults"):
                    setattr(node, "defaults", {})
                node.defaults["stores"] = self.stores

        # Run the pipeline
        super().run(data=data, debug=debug)


def load_pipelines(path: Path, _reader: Optional[Callable[..., Any]] = None):
    return load_canals_pipelines(path=path, _reader=_reader)


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer: Optional[Callable[..., Any]] = None):
    save_canals_pipelines(pipelines=pipelines, path=path, _writer=_writer)
