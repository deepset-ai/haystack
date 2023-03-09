from typing import List, Dict, Union, Any, Tuple, Optional

from canals import Pipeline as CanalsPipeline, PipelineError


class NoSuchStoreError(PipelineError):
    pass


class Pipeline(CanalsPipeline):
    """
    Haystack Pipeline is a thin wrapper over Canals' Pipelines to add support for Stores.
    """

    def __init__(self):
        super().__init__()
        self.stores: Dict[str, object] = {}

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

    def run(
        self,
        data: Union[Dict[str, Any], List[Tuple[str, Any]]],
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        debug: bool = False,
    ):
        """
        Wrapper on top of Canals Pipeline.run(). Adds the `stores` parameter to all nodes.
        """
        if not parameters:
            parameters = {}

        for node in self.graph.nodes:
            if not node in parameters.keys():
                parameters[node] = {"stores": self.stores}
            else:
                parameters[node] = {"stores": self.stores, **parameters[node]}

        super().run(data=data, parameters=parameters, debug=debug)
