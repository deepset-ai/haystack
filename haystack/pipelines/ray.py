from __future__ import annotations
import inspect
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
    validate_config,
)
from haystack.nodes.base import BaseComponent, RootNode
from haystack.pipelines.base import Pipeline


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
        ray.init(address=address, **ray_args)
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
        return RayPipeline.load_from_config(
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
        cls, component_name: str, pipeline_config: dict, serve_deployment_kwargs: Optional[Dict[str, Any]] = {}
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
                outgoing_edges_input_node = self.graph.nodes[input_node_name]["component"].outgoing_edges
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

    def _get_run_node_signature(self, node_id: str):
        return inspect.signature(self.graph.nodes[node_id]["component"].remote).parameters.keys()


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
