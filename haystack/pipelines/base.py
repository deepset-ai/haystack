import copy
import inspect
import logging
import os
import traceback
from pathlib import Path
from typing import List, Optional, Any

import networkx as nx
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack.schema import MultiLabel, Document
from haystack.nodes.base import BaseComponent
from haystack.document_store.base import BaseDocumentStore


logger = logging.getLogger(__name__)


class RootNode(BaseComponent):
    """
    RootNode feeds inputs together with corresponding params to a Pipeline.
    """
    outgoing_edges = 1

    def run(self, root_node: str):  # type: ignore
        return {}, "output_1"


class BasePipeline:
    """
    Base class for pipelines, providing the most basic methods to load and save them in different ways. 
    See also the `Pipeline` class for the actual pipeline logic.
    """
    def run(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```yaml
            |   version: '0.8'
            |
            |    components:    # define all the building-blocks for Pipeline
            |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
            |      type: FARMReader    # Haystack Class name for the component
            |      params:
            |        no_ans_boost: -10
            |        model_name_or_path: deepset/roberta-base-squad2
            |    - name: MyESRetriever
            |      type: ElasticsearchRetriever
            |      params:
            |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
            |        custom_query: null
            |    - name: MyDocumentStore
            |      type: ElasticsearchDocumentStore
            |      params:
            |        index: haystack_test
            |
            |    pipelines:    # multiple Pipelines can be defined using the components from above
            |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
            |      nodes:
            |      - name: MyESRetriever
            |        inputs: [Query]
            |      - name: MyReader
            |        inputs: [MyESRetriever]
            ```

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        pipeline_config = cls._get_pipeline_config_from_yaml(path=path, pipeline_name=pipeline_name)
        if pipeline_config["type"] == "Pipeline":
            return Pipeline.load_from_yaml(
                path=path, pipeline_name=pipeline_name, overwrite_with_env_variables=overwrite_with_env_variables
            )
        elif pipeline_config["type"] == "RayPipeline":
            return RayPipeline.load_from_yaml(
                path=path, pipeline_name=pipeline_name, overwrite_with_env_variables=overwrite_with_env_variables
            )
        else:
            raise KeyError(f"Pipeline Type '{pipeline_config['type']}' is not a valid. The available types are"
                           f"'Pipeline' and 'RayPipeline'.")

    @classmethod
    def _get_pipeline_config_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None):
        """
        Get the definition of Pipeline from a given YAML. If the YAML contains more than one Pipeline,
        then the pipeline_name must be supplied.

        :param path: Path of Pipeline YAML file.
        :param pipeline_name: name of the Pipeline.
        """
        with open(path, "r", encoding='utf-8') as stream:
            data = yaml.safe_load(stream)

        if pipeline_name is None:
            if len(data["pipelines"]) == 1:
                pipeline_config = data["pipelines"][0]
            else:
                raise Exception("The YAML contains multiple pipelines. Please specify the pipeline name to load.")
        else:
            pipelines_in_yaml = list(filter(lambda p: p["name"] == pipeline_name, data["pipelines"]))
            if not pipelines_in_yaml:
                raise KeyError(f"Cannot find any pipeline with name '{pipeline_name}' declared in the YAML file.")
            pipeline_config = pipelines_in_yaml[0]

        return pipeline_config

    @classmethod
    def _read_yaml(cls, path: Path, pipeline_name: Optional[str], overwrite_with_env_variables: bool):
        """
        Parse the YAML and return the full YAML config, pipeline_config, and definitions of all components.

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        pipeline_config = cls._get_pipeline_config_from_yaml(path=path, pipeline_name=pipeline_name)

        definitions = {}  # definitions of each component from the YAML.
        component_definitions = copy.deepcopy(data["components"])
        for definition in component_definitions:
            if overwrite_with_env_variables:
                cls._overwrite_with_env_variables(definition)
            name = definition.pop("name")
            definitions[name] = definition

        return data, pipeline_config, definitions

    @classmethod
    def _overwrite_with_env_variables(cls, definition: dict):
        """
        Overwrite the YAML configuration with environment variables. For example, to change index name param for an
        ElasticsearchDocumentStore, an env variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
        `_` sign must be used to specify nested hierarchical properties.

        :param definition: a dictionary containing the YAML definition of a component.
        """
        env_prefix = f"{definition['name']}_params_".upper()
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                param_name = key.replace(env_prefix, "").lower()
                definition["params"][param_name] = value


class Pipeline(BasePipeline):
    """
    Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

    Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
    flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
    Reader from multiple Retrievers, or re-ranking of candidate documents.
    """

    def __init__(self):
        self.graph = DiGraph()
        self.root_node = None
        self.components: dict = {}

    def add_node(self, component, name: str, inputs: List[str]):
        """
        Add a new node to the pipeline.

        :param component: The object to be called when the data is passed to the node. It can be a Haystack component
                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                          method to process incoming data from predecessor node.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
                       edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        if self.root_node is None:
            root_node = inputs[0]
            if root_node in ["Query", "File"]:
                self.root_node = root_node
                self.graph.add_node(root_node, component=RootNode())
            else:
                raise KeyError(f"Root node '{root_node}' is invalid. Available options are 'Query' and 'File'.")
        component.name = name
        self.graph.add_node(name, component=component, inputs=inputs)

        if len(self.graph.nodes) == 2:  # first node added; connect with Root
            assert len(inputs) == 1 and inputs[0].split(".")[0] == self.root_node, \
                f"The '{name}' node can only input from {self.root_node}. " \
                f"Set the 'inputs' parameter to ['{self.root_node}']"
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
                outgoing_edges_input_node = self.graph.nodes[i]["component"].outgoing_edges
                assert outgoing_edges_input_node == 1, (
                    f"Adding an edge from {i} to {name} is ambiguous as {i} has {outgoing_edges_input_node} edges. "
                    f"Please specify the output explicitly."
                )
                input_node_name = i
                input_edge_name = "output_1"
            self.graph.add_edge(input_node_name, name, label=input_edge_name)

    def get_node(self, name: str) -> Optional[BaseComponent]:
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        graph_node = self.graph.nodes.get(name)
        component = graph_node["component"] if graph_node else None
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.graph.nodes[name]["component"] = component

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
        debug_logs: Optional[bool] = None
    ):
        """
            Runs the pipeline, one node at a time.

            :param query: The search query (for query pipelines only)
            :param file_paths: The files to index (for indexing pipelines only)
            :param labels: 
            :param documents:
            :param meta:
            :param params: Dictionary of parameters to be dispatched to the nodes. 
                           If you want to pass a param to all nodes, you can just use: {"top_k":10}
                           If you want to pass it to targeted nodes, you can do:
                           {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
            :param debug: Whether the pipeline should instruct nodes to collect debug information
                          about their execution. By default these include the input parameters
                          they received, the output they generated, and eventual logs (of any severity)
                          emitted. All debug information can then be found in the dict returned
                          by this method under the key "_debug"
            :param debug_logs: Whether all the logs of the node should be printed in the console,
                               regardless of their severity and of the existing logger's settings.
        """
        # validate the node names
        if params:
            if not all(node_id in self.graph.nodes for node_id in params.keys()):

                # Might be a non-targeted param. Verify that too
                not_a_node = set(params.keys()) - set(self.graph.nodes)
                valid_global_params = set()
                for node_id in self.graph.nodes:
                    run_signature_args = inspect.signature(self.graph.nodes[node_id]["component"].run).parameters.keys()
                    valid_global_params |= set(run_signature_args)
                invalid_keys = [key for key in not_a_node if key not in valid_global_params]

                if invalid_keys:
                    raise ValueError(f"No node(s) or global parameter(s) named {', '.join(invalid_keys)} found in pipeline.")

        node_output = None
        queue = {
            self.root_node: {"root_node": self.root_node, "params": params}
        }  # ordered dict with "node_id" -> "input" mapping that acts as a FIFO queue
        if query:
            queue[self.root_node]["query"] = query
        if file_paths:
            queue[self.root_node]["file_paths"] = file_paths
        if labels:
            queue[self.root_node]["labels"] = labels
        if documents:
            queue[self.root_node]["documents"] = documents
        if meta:
            queue[self.root_node]["meta"] = meta

        i = 0  # the first item is popped off the queue unless it is a "join" node with unprocessed predecessors
        while queue:
            node_id = list(queue.keys())[i]
            node_input = queue[node_id]
            node_input["node_id"] = node_id

            # Apply debug attributes to the node input params
            # NOTE: global debug attributes will override the value specified
            # in each node's params dictionary.
            if debug is not None:
                if node_id not in node_input["params"].keys():
                    node_input["params"][node_id] = {}
                node_input["params"][node_id]["debug"] = debug
                if debug_logs is not None:
                    node_input["params"][node_id]["debug_logs"] = debug_logs

            predecessors = set(nx.ancestors(self.graph, node_id))
            if predecessors.isdisjoint(set(queue.keys())):  # only execute if predecessor nodes are executed
                try:
                    logger.debug(f"Running node `{node_id}` with input `{node_input}`")
                    node_output, stream_id = self.graph.nodes[node_id]["component"]._dispatch_run(**node_input)
                except Exception as e:
                    tb = traceback.format_exc()
                    raise Exception(f"Exception while running node `{node_id}` with input `{node_input}`: {e}, full stack trace: {tb}")
                queue.pop(node_id)
                next_nodes = self.get_next_nodes(node_id, stream_id)
                for n in next_nodes:  # add successor nodes with corresponding inputs to the queue
                    if queue.get(n):  # concatenate inputs if it's a join node
                        existing_input = queue[n]
                        if "inputs" not in existing_input.keys():
                            updated_input: dict = {"inputs": [existing_input, node_output], "params": params}
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
        return node_output

    def get_next_nodes(self, node_id: str, stream_id: str):
        current_node_edges = self.graph.edges(node_id, data=True)
        next_nodes = [
            next_node
            for _, next_node, data in current_node_edges
            if not stream_id or data["label"] == stream_id or stream_id == "output_all"
        ]
        return next_nodes

    def get_nodes_by_class(self, class_type) -> List[Any]:
        """
        Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
        This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
        Example:
        | from haystack.document_store.base import BaseDocumentStore
        | INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
        | res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

        :return: List of components that are an instance the requested class
        """

        matches = [self.graph.nodes.get(node)["component"]
                   for node in self.graph.nodes
                   if isinstance(self.graph.nodes.get(node)["component"], class_type)]
        return matches

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        """
        Return the document store object used in the current pipeline.

        :return: Instance of DocumentStore or None
        """
        matches = self.get_nodes_by_class(class_type=BaseDocumentStore)
        if len(matches) > 1:
            raise Exception(f"Multiple Document Stores found in Pipeline: {matches}")
        elif len(matches) == 0:
            return None
        else:
            return matches[0]

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        try:
            import pygraphviz
        except ImportError:
            raise ImportError(f"Could not import `pygraphviz`. Please install via: \n"
                              f"pip install pygraphviz\n"
                              f"(You might need to run this first: apt install libgraphviz-dev graphviz )")

        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw(path)

    @classmethod
    def load_from_yaml(cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```yaml
            |   version: '0.8'
            |
            |    components:    # define all the building-blocks for Pipeline
            |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
            |      type: FARMReader    # Haystack Class name for the component
            |      params:
            |        no_ans_boost: -10
            |        model_name_or_path: deepset/roberta-base-squad2
            |    - name: MyESRetriever
            |      type: ElasticsearchRetriever
            |      params:
            |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
            |        custom_query: null
            |    - name: MyDocumentStore
            |      type: ElasticsearchDocumentStore
            |      params:
            |        index: haystack_test
            |
            |    pipelines:    # multiple Pipelines can be defined using the components from above
            |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
            |      nodes:
            |      - name: MyESRetriever
            |        inputs: [Query]
            |      - name: MyReader
            |        inputs: [MyESRetriever]
            ```

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        data, pipeline_config, definitions = cls._read_yaml(
            path=path, pipeline_name=pipeline_name, overwrite_with_env_variables=overwrite_with_env_variables
        )

        pipeline = cls()

        components: dict = {}  # instances of component objects.
        for node_config in pipeline_config["nodes"]:
            name = node_config["name"]
            component = cls._load_or_get_component(name=name, definitions=definitions, components=components)
            pipeline.add_node(component=component, name=node_config["name"], inputs=node_config.get("inputs", []))

        return pipeline

    @classmethod
    def _load_or_get_component(cls, name: str, definitions: dict, components: dict):
        """
        Load a component from the definition or return if component object already present in `components` dict.

        :param name: name of the component to load or get.
        :param definitions: dict containing definitions of all components retrieved from the YAML.
        :param components: dict containing component objects.
        """
        try:
            if name in components.keys():  # check if component is already loaded.
                return components[name]

            component_params = definitions[name].get("params", {})
            component_type = definitions[name]["type"]
            logger.debug(f"Loading component `{name}` of type `{definitions[name]['type']}`")

            for key, value in component_params.items():
                # Component params can reference to other components. For instance, a Retriever can reference a
                # DocumentStore defined in the YAML. All references should be recursively resolved.
                if isinstance(value, str) and value in definitions.keys():  # check if the param value is a reference to another component.
                    if value not in components.keys():  # check if the referenced component is already loaded.
                        cls._load_or_get_component(name=value, definitions=definitions, components=components)
                    component_params[key] = components[value]  # substitute reference (string) with the component object.

            instance = BaseComponent.load_from_args(component_type=component_type, **component_params)
            components[name] = instance
        except Exception as e:
            raise Exception(f"Failed loading pipeline component '{name}': {e}")
        return instance

    def save_to_yaml(self, path: Path, return_defaults: bool = False):
        """
        Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

        :param path: path of the output YAML file.
        :param return_defaults: whether to output parameters that have the default values.
        """
        nodes = self.graph.nodes

        pipeline_name = self.root_node.lower()
        pipelines: dict = {pipeline_name: {"name": pipeline_name, "type": "Pipeline", "nodes": []}}

        components = {}
        for node in nodes:
            if node == self.root_node:
                continue
            component_instance = self.graph.nodes.get(node)["component"]
            component_type = component_instance.pipeline_config["type"]
            component_params = component_instance.pipeline_config["params"]
            components[node] = {"name": node, "type": component_type, "params": {}}
            component_signature = inspect.signature(type(component_instance)).parameters
            for key, value in component_params.items():
                # A parameter for a Component could be another Component. For instance, a Retriever has
                # the DocumentStore as a parameter.
                # Component configs must be a dict with a "type" key. The "type" keys distinguishes between
                # other parameters like "custom_mapping" that are dicts.
                # This currently only checks for the case single-level nesting case, wherein, "a Component has another
                # Component as a parameter". For deeper nesting cases, this function should be made recursive.
                if isinstance(value, dict) and "type" in value.keys():  # the parameter is a Component
                    components[node]["params"][key] = value["type"]
                    sub_component_signature = inspect.signature(BaseComponent.subclasses[value["type"]]).parameters
                    params = {
                        k: v for k, v in value["params"].items()
                        if sub_component_signature[k].default != v or return_defaults is True
                    }
                    components[value["type"]] = {"name": value["type"], "type": value["type"], "params": params}
                else:
                    if component_signature[key].default != value or return_defaults is True:
                        components[node]["params"][key] = value

            # create the Pipeline definition with how the Component are connected
            pipelines[pipeline_name]["nodes"].append({"name": node, "inputs": list(self.graph.predecessors(node))})

        config = {"components": list(components.values()), "pipelines": list(pipelines.values()), "version": "0.8"}

        with open(path, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
