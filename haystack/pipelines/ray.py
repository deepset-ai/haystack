from __future__ import annotations
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import networkx as nx

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

from haystack.errors import PipelineError
from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
    validate_config,
)
from haystack.nodes.base import BaseComponent, RootNode
from haystack.pipelines.base import Pipeline
from haystack.schema import Document, MultiLabel


logger = logging.getLogger(__name__)


class RayPipeline(Pipeline):
    """
    [Ray](https://ray.io) is a framework for distributed computing.

    With Ray, you can distribute a Pipeline's components across a cluster of machines. The individual components of a
    Pipeline can be independently scaled. For instance, an extractive QA Pipeline deployment can have three replicas
    of the Reader and a single replica for the Retriever. This way, you can use your resources more efficiently by horizontally scaling Components.

    To set the number of replicas, add  `num_replicas` in the YAML configuration for the node in a pipeline:

    ```yaml
    components:
        ...

    pipelines:
        - name: ray_query_pipeline
          type: RayPipeline
          nodes:
            - name: Retriever
              inputs: [ Query ]
              serve_deployment_kwargs:
                num_replicas: 2  # number of replicas to create on the Ray cluster
    ```

    A Ray Pipeline can only be created with a YAML Pipeline configuration.

    ```python
    from haystack.pipeline import RayPipeline
    pipeline = RayPipeline.load_from_yaml(path="my_pipelines.yaml", pipeline_name="my_query_pipeline")
    pipeline.run(query="What is the capital of Germany?")
    ```

    By default, RayPipelines create an instance of RayServe locally. To connect to an existing Ray instance,
    set the `address` parameter when creating the RayPipeline instance.

    YAML definitions of Ray pipelines are validated at load. For more information, see [YAML File Definitions](https://haystack-website-git-fork-fstau-dev-287-search-deepset-overnice.vercel.app/components/pipelines#yaml-file-definitions).
    """

    def __init__(
        self,
        address: Optional[str] = None,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
    ):
        """
        :param address: The IP address for the Ray cluster. If set to `None`, a local Ray instance is started.
        :param ray_args: Optional parameters for initializing Ray.
        :param serve_args: Optional parameters for initializing Ray Serve.
        """
        ray_args = ray_args or {}
        if not ray.is_initialized():
            ray.init(address=address, **ray_args)
        else:
            logger.warning("Ray was already initialized, so reusing that for this RayPipeline.")
        self._serve_controller_client = serve.start(**serve_args)
        super().__init__()

    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
        address: Optional[str] = None,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
    ):
        validate_config(pipeline_config, strict_version_check=strict_version_check, extras="ray")

        pipeline_definition = get_pipeline_definition(pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config, overwrite_with_env_variables=overwrite_with_env_variables
        )
        pipeline = cls(address=address, ray_args=ray_args or {}, serve_args=serve_args or {})

        for node_config in pipeline_definition["nodes"]:
            if pipeline.root_node is None:
                root_node = node_config["inputs"][0]
                if root_node in ["Query", "File"]:
                    handle = cls._create_ray_deployment(component_name=root_node, pipeline_config=pipeline_config)
                    pipeline._add_ray_deployment_in_graph(handle=handle, name=root_node, outgoing_edges=1, inputs=[])
                else:
                    raise KeyError(f"Root node '{root_node}' is invalid. Available options are 'Query' and 'File'.")

            name = node_config["name"]
            component_type = component_definitions[name]["type"]
            component_class = BaseComponent.get_subclass(component_type)
            serve_deployment_kwargs = next(node for node in pipeline_definition["nodes"] if node["name"] == name).get(
                "serve_deployment_kwargs", {}
            )
            handle = cls._create_ray_deployment(
                component_name=name, pipeline_config=pipeline_config, serve_deployment_kwargs=serve_deployment_kwargs
            )
            pipeline._add_ray_deployment_in_graph(
                handle=handle,
                name=name,
                outgoing_edges=component_class.outgoing_edges,
                inputs=node_config.get("inputs", []),
            )

        return pipeline

    @classmethod
    def load_from_yaml(  # type: ignore
        cls,
        path: Path,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        address: Optional[str] = None,
        strict_version_check: bool = False,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

           ```yaml
           version: '1.0.0'

            components:    # define all the building-blocks for Pipeline
            - name: MyReader       # custom-name for the component; helpful for visualization & debugging
              type: FARMReader    # Haystack Class name for the component
              params:
                no_ans_boost: -10
                model_name_or_path: deepset/roberta-base-squad2
            - name: MyRetriever
              type: BM25Retriever
              params:
                document_store: MyDocumentStore    # params can reference other components defined in the YAML
                custom_query: null
            - name: MyDocumentStore
              type: ElasticsearchDocumentStore
              params:
                index: haystack_test

            pipelines:    # multiple Pipelines can be defined using the components from above
            - name: my_query_pipeline    # a simple extractive-qa Pipeline
              type: RayPipeline
              nodes:
              - name: MyRetriever
                inputs: [Query]
                serve_deployment_kwargs:
                  num_replicas: 2    # number of replicas to create on the Ray cluster
              - name: MyReader
                inputs: [MyRetriever]
           ```


        Note that, in case of a mismatch in version between Haystack and the YAML, a warning will be printed.
        If the pipeline loads correctly regardless, save again the pipeline using `RayPipeline.save_to_yaml()` to remove the warning.

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        :param address: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
        :param serve_args: Optional parameters for initializing Ray Serve.
        """
        pipeline_config = read_pipeline_config_from_yaml(path)
        return cls.load_from_config(
            pipeline_config=pipeline_config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
            strict_version_check=strict_version_check,
            address=address,
            ray_args=ray_args,
            serve_args=serve_args,
        )

    @classmethod
    def _create_ray_deployment(
        cls, component_name: str, pipeline_config: dict, serve_deployment_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Create a Ray Deployment for the Component.

        :param component_name: Class name of the Haystack Component.
        :param pipeline_config: The Pipeline config YAML parsed as a dict.
        :param serve_deployment_kwargs: An optional dictionary of arguments to be supplied to the
                                        `ray.serve.deployment()` method, like `num_replicas`, `ray_actor_options`,
                                        `max_concurrent_queries`, etc. See potential values in the
                                         Ray Serve API docs (https://docs.ray.io/en/latest/serve/package-ref.html)
                                         under the `ray.serve.deployment()` method
        """
        if serve_deployment_kwargs is None:
            serve_deployment_kwargs = {}
        RayDeployment = serve.deployment(
            _RayDeploymentWrapper, name=component_name, **serve_deployment_kwargs  # type: ignore
        )
        RayDeployment.deploy(pipeline_config, component_name)
        handle = RayDeployment.get_handle()
        return handle

    def add_node(self, component, name: str, inputs: List[str]):
        raise NotImplementedError(
            "The current implementation of RayPipeline only supports loading Pipelines from a YAML file."
        )

    def _add_ray_deployment_in_graph(self, handle, name: str, outgoing_edges: int, inputs: List[str]):
        """
        Add the Ray deployment handle in the Pipeline Graph.

        :param handle: Ray deployment `handle` to add in the Pipeline Graph. The handle allow calling a Ray deployment
                       from Python: https://docs.ray.io/en/main/serve/package-ref.html#servehandle-api.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
                       edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        self.graph.add_node(name, component=handle, inputs=inputs, outgoing_edges=outgoing_edges)

        if len(self.graph.nodes) == 2:  # first node added; connect with Root
            self.graph.add_edge(self.root_node, name, label="output_1")
            return

        for i in inputs:
            if "." in i:
                [input_node_name, input_edge_name] = i.split(".")
                assert "output_" in input_edge_name, f"'{input_edge_name}' is not a valid edge name."
                outgoing_edges_input_node = self.graph.nodes[input_node_name]["outgoing_edges"]
                assert int(input_edge_name.split("_")[1]) <= outgoing_edges_input_node, (
                    f"Cannot connect '{input_edge_name}' from '{input_node_name}' as it only has "
                    f"{outgoing_edges_input_node} outgoing edge(s)."
                )
            else:
                outgoing_edges_input_node = self.graph.nodes[i]["outgoing_edges"]
                assert outgoing_edges_input_node == 1, (
                    f"Adding an edge from {i} to {name} is ambiguous as {i} has {outgoing_edges_input_node} edges. "
                    f"Please specify the output explicitly."
                )
                input_node_name = i
                input_edge_name = "output_1"
            self.graph.add_edge(input_node_name, name, label=input_edge_name)

    def _run_node(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:
        return ray.get(self.graph.nodes[node_id]["component"].remote(**node_input))

    async def _run_node_async(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:
        # Async calling of Ray Deployments instead of using `ray.get()` as it is done
        # in the sync version, in `_run_node()` above.
        # See https://docs.ray.io/en/latest/ray-core/actors/async_api.html#objectrefs-as-asyncio-futures
        return await self.graph.nodes[node_id]["component"].remote(**node_input)

    def _get_run_node_signature(self, node_id: str):
        return inspect.signature(self.graph.nodes[node_id]["component"].remote).parameters.keys()

    # async version of the `Pipeline.run()` method
    async def run_async(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[Union[dict, List[dict]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        """
        Runs the Pipeline, one node at a time.

        :param query: The search query (for query pipelines only).
        :param file_paths: The files to index (for indexing pipelines only).
        :param labels: Ground-truth labels that you can use to perform an isolated evaluation of pipelines. These labels are input to nodes in the pipeline.
        :param documents: A list of Document objects to be processed by the Pipeline Nodes.
        :param meta: Files' metadata. Used in indexing pipelines in combination with `file_paths`.
        :param params: A dictionary of parameters that you want to pass to the nodes.
                       To pass a parameter to all Nodes, use: `{"top_k": 10}`.
                       To pass a parameter to targeted Nodes, run:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}`
        :param debug: Specifies whether the Pipeline should instruct Nodes to collect debug information
                      about their execution. By default, this information includes the input parameters
                      the Nodes received and the output they generated. You can then find all debug information in the dictionary returned by this method under the key `_debug`.
        """
        # validate the node names
        self._validate_node_names_in_params(params=params)

        root_node = self.root_node
        if not root_node:
            raise PipelineError("Cannot run a pipeline with no nodes.")

        node_output = None
        queue: Dict[str, Any] = {
            root_node: {"root_node": root_node, "params": params}
        }  # ordered dict with "node_id" -> "input" mapping that acts as a FIFO queue
        if query is not None:
            queue[root_node]["query"] = query
        if file_paths:
            queue[root_node]["file_paths"] = file_paths
        if labels:
            queue[root_node]["labels"] = labels
        if documents:
            queue[root_node]["documents"] = documents
        if meta:
            queue[root_node]["meta"] = meta

        i = 0  # the first item is popped off the queue unless it is a "join" node with unprocessed predecessors
        while queue:
            node_id = list(queue.keys())[i]
            node_input = queue[node_id]
            node_input["node_id"] = node_id

            # Apply debug attributes to the node input params
            # NOTE: global debug attributes will override the value specified
            # in each node's params dictionary.
            if debug is None and node_input:
                if node_input.get("params", {}):
                    debug = params.get("debug", None)  # type: ignore
            if debug is not None:
                if not node_input.get("params", None):
                    node_input["params"] = {}
                if node_id not in node_input["params"].keys():
                    node_input["params"][node_id] = {}
                node_input["params"][node_id]["debug"] = debug

            predecessors = set(nx.ancestors(self.graph, node_id))
            if predecessors.isdisjoint(set(queue.keys())):  # only execute if predecessor nodes are executed
                try:
                    logger.debug("Running node '%s` with input: %s", node_id, node_input)
                    node_output, stream_id = await self._run_node_async(node_id, node_input)
                except Exception as e:
                    # The input might be a really large object with thousands of embeddings.
                    # If you really want to see it, raise the log level.
                    logger.debug("Exception while running node '%s' with input %s", node_id, node_input)
                    raise Exception(
                        f"Exception while running node '{node_id}': {e}\nEnable debug logging to see the data that was passed when the pipeline failed."
                    ) from e
                queue.pop(node_id)
                #
                if stream_id == "split":
                    for stream_id in [key for key in node_output.keys() if key.startswith("output_")]:
                        current_node_output = {k: v for k, v in node_output.items() if not k.startswith("output_")}
                        current_docs = node_output.pop(stream_id)
                        current_node_output["documents"] = current_docs
                        next_nodes = self.get_next_nodes(node_id, stream_id)
                        for n in next_nodes:
                            queue[n] = current_node_output
                else:
                    next_nodes = self.get_next_nodes(node_id, stream_id)
                    for n in next_nodes:  # add successor nodes with corresponding inputs to the queue
                        if queue.get(n):  # concatenate inputs if it's a join node
                            existing_input = queue[n]
                            if "inputs" not in existing_input.keys():
                                updated_input: dict = {"inputs": [existing_input, node_output], "params": params}
                                if "_debug" in existing_input.keys() or "_debug" in node_output.keys():
                                    updated_input["_debug"] = {
                                        **existing_input.get("_debug", {}),
                                        **node_output.get("_debug", {}),
                                    }
                                if query:
                                    updated_input["query"] = query
                                if file_paths:
                                    updated_input["file_paths"] = file_paths
                                if labels:
                                    updated_input["labels"] = labels
                                if documents:
                                    updated_input["documents"] = documents
                                if meta:
                                    updated_input["meta"] = meta
                            else:
                                existing_input["inputs"].append(node_output)
                                updated_input = existing_input
                            queue[n] = updated_input
                        else:
                            queue[n] = node_output
                i = 0
            else:
                i += 1  # attempt executing next node in the queue as current `node_id` has unprocessed predecessors

        self.run_total += 1
        # Disabled due to issue https://github.com/deepset-ai/haystack/issues/3970
        # self.send_pipeline_event_if_needed(is_indexing=file_paths is not None)
        return node_output

    def send_pipeline_event(self, is_indexing: bool = False):
        """To avoid the RayPipeline serialization bug described at
        https://github.com/deepset-ai/haystack/issues/3970"""
        pass


class _RayDeploymentWrapper:
    """
    Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

    In case of Haystack, some Components like Retrievers have complex init methods that needs objects
    like Document Stores.

    This wrapper class encapsulates the initialization of Components. Given a Component Class
    name, it creates an instance using the YAML Pipeline config.
    """

    node: BaseComponent

    def __init__(self, pipeline_config: dict, component_name: str):
        """
        Create an instance of Component.

        :param pipeline_config: Pipeline YAML parsed as a dict.
        :param component_name: Component Class name.
        """
        if component_name in ["Query", "File"]:
            self.node = RootNode()
        else:
            self.node = self.load_from_pipeline_config(pipeline_config, component_name)

    def __call__(self, *args, **kwargs):
        """
        Ray calls this method which is then re-directed to the corresponding component's run().
        """
        return self.node._dispatch_run(*args, **kwargs)

    @staticmethod
    def load_from_pipeline_config(pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        all_component_configs = pipeline_config["components"]
        all_component_names = [comp["name"] for comp in all_component_configs]
        component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
        component_params = component_config["params"]

        for key, value in component_params.items():
            if value in all_component_names:  # check if the param value is a reference to another component
                component_params[key] = _RayDeploymentWrapper.load_from_pipeline_config(pipeline_config, value)

        component_instance = BaseComponent._create_instance(
            component_type=component_config["type"], component_params=component_params, name=component_name
        )
        return component_instance
