import copy
import inspect
import logging
import os
import traceback
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict, Union, Any
import pickle
import urllib
from functools import wraps

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore


from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

import networkx as nx
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack import BaseComponent, MultiLabel, Document
from haystack.generator.base import BaseGenerator
from haystack.document_store.base import BaseDocumentStore
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.summarizer.base import BaseSummarizer
from haystack.translator.base import BaseTranslator
from haystack.document_store.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class BasePipeline:

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
    ):
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


class BaseStandardPipeline(ABC):
    pipeline: Pipeline

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

        self.pipeline.add_node(component=component, name=name, inputs=inputs)

    def get_node(self, name: str):
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        component = self.pipeline.get_node(name)
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.pipeline.set_node(name, component)

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        self.pipeline.draw(path)


class ExtractiveQAPipeline(BaseStandardPipeline):
    def __init__(self, reader: BaseReader, retriever: BaseRetriever):
        """
        Initialize a Pipeline for Extractive Question Answering.

        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance,
                       params={"retriever": {"top_k": 10}, "reader": {"top_k": 5}}
        """
        output = self.pipeline.run(query=query, params=params)
        return output


class DocumentSearchPipeline(BaseStandardPipeline):
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize a Pipeline for semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, query: str, params: Optional[dict] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"retriever": {"top_k": 10}}
        """
        output = self.pipeline.run(query=query, params=params)
        document_dicts = [doc.to_dict() for doc in output["documents"]]
        output["documents"] = document_dicts
        return output


class GenerativeQAPipeline(BaseStandardPipeline):
    def __init__(self, generator: BaseGenerator, retriever: BaseRetriever):
        """
        Initialize a Pipeline for Generative Question Answering.

        :param generator: Generator instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=generator, name="Generator", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `generator`. For instance,
                       params={"retriever": {"top_k": 10}, "generator": {"top_k": 5}}
        """
        output = self.pipeline.run(query=query, params=params)
        return output


class SearchSummarizationPipeline(BaseStandardPipeline):
    def __init__(self, summarizer: BaseSummarizer, retriever: BaseRetriever, return_in_answer_format: bool = False):
        """
        Initialize a Pipeline that retrieves documents for a query and then summarizes those documents.

        :param summarizer: Summarizer instance
        :param retriever: Retriever instance
        :param return_in_answer_format: Whether the results should be returned as documents (False) or in the answer
                                        format used in other QA pipelines (True). With the latter, you can use this
                                        pipeline as a "drop-in replacement" for other QA pipelines.
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=summarizer, name="Summarizer", inputs=["Retriever"])
        self.return_in_answer_format = return_in_answer_format

    def run(self, query: str, params: Optional[dict] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `summarizer`. For instance,
                       params={"retriever": {"top_k": 10}, "summarizer": {"generate_single_summary": True}}
                """
        output = self.pipeline.run(query=query, params=params)

        # Convert to answer format to allow "drop-in replacement" for other QA pipelines
        if self.return_in_answer_format:
            results: Dict = {"query": query, "answers": []}
            docs = deepcopy(output["documents"])
            for doc in docs:
                cur_answer = {
                    "query": query,
                    "answer": doc.text,
                    "document_id": doc.id,
                    "context": doc.meta.pop("context"),
                    "score": None,
                    "offset_start": None,
                    "offset_end": None,
                    "meta": doc.meta,
                }

                results["answers"].append(cur_answer)
        else:
            results = output
        return results


class FAQPipeline(BaseStandardPipeline):
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize a Pipeline for finding similar FAQs using semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, query: str, params: Optional[dict] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever`. For instance, params={"retriever": {"top_k": 10}}
        """
        output = self.pipeline.run(query=query, params=params)
        documents = output["documents"]

        results: Dict = {"query": query, "answers": []}
        for doc in documents:
            # TODO proper calibration of pseudo probabilities
            cur_answer = {
                "query": doc.text,
                "answer": doc.meta["answer"],
                "document_id": doc.id,
                "context": doc.meta["answer"],
                "score": doc.score,
                "offset_start": 0,
                "offset_end": len(doc.meta["answer"]),
                "meta": doc.meta,
            }

            results["answers"].append(cur_answer)
        return results


class TranslationWrapperPipeline(BaseStandardPipeline):

    """
    Takes an existing search pipeline and adds one "input translation node" after the Query and one
    "output translation" node just before returning the results
    """

    def __init__(
        self,
        input_translator: BaseTranslator,
        output_translator: BaseTranslator,
        pipeline: BaseStandardPipeline
    ):
        """
        Wrap a given `pipeline` with the `input_translator` and `output_translator`.

        :param input_translator: A Translator node that shall translate the input query from language A to B
        :param output_translator: A Translator node that shall translate the pipeline results from language B to A
        :param pipeline: The pipeline object (e.g. ExtractiveQAPipeline) you want to "wrap".
                         Note that pipelines with split or merge nodes are currently not supported.
        """

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=input_translator, name="InputTranslator", inputs=["Query"])

        graph = pipeline.pipeline.graph
        previous_node_name = ["InputTranslator"]
        # Traverse in BFS
        for node in graph.nodes:
            if node == "Query":
                continue

            # TODO: Do not work properly for Join Node and Answer format
            if graph.nodes[node]["inputs"] and len(graph.nodes[node]["inputs"]) > 1:
                raise AttributeError("Split and merge nodes are not supported currently")

            self.pipeline.add_node(name=node, component=graph.nodes[node]["component"], inputs=previous_node_name)
            previous_node_name = [node]

        self.pipeline.add_node(component=output_translator, name="OutputTranslator", inputs=previous_node_name)

    def run(self, **kwargs):
        output = self.pipeline.run(**kwargs)
        return output


class QuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes documents as input and generates
    questions that it thinks can be answered by the documents.
    """
    def __init__(self, question_generator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])

    def run(self, documents, params: Optional[dict] = None):
        output = self.pipeline.run(documents=documents, params=params)
        return output


class RetrieverQuestionGenerationPipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes a query as input, performs retrieval, and then generates
    questions that it thinks can be answered by the retrieved documents.
    """
    def __init__(self, retriever, question_generator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=question_generator, name="Question Generator", inputs=["Retriever"])

    def run(self, query, params: Optional[dict] = None):
        output = self.pipeline.run(query=query, params=params)
        return output


class QuestionAnswerGenerationPipeline(BaseStandardPipeline):
    """
    This is a pipeline which takes a document as input, generates questions that the model thinks can be answered by
    this document, and then performs question answering of this questions using that single document.
    """
    def __init__(self, question_generator, reader):
        question_generator.run = self.formatting_wrapper(question_generator.run)
        # Overwrite reader.run function so it can handle a batch of questions being passed on by the QuestionGenerator
        reader.run = reader.run_batch
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["QuestionGenerator"])

    # This is used to format the output of the QuestionGenerator so that its questions are ready to be answered by the reader
    def formatting_wrapper(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            output, output_stream = fn(*args, **kwargs)
            questions = output["generated_questions"][0]["questions"]
            documents = output["documents"]
            query_doc_list = []
            for q in questions:
                query_doc_list.append({"queries": q, "docs": documents})
            kwargs["query_doc_list"] = query_doc_list
            return kwargs, output_stream
        return wrapper

    def run(self, documents: List[Document], params: Optional[dict] = None):  # type: ignore
        output = self.pipeline.run(documents=documents, params=params)
        return output


class RootNode(BaseComponent):
    """
    RootNode feeds inputs together with corresponding params to a Pipeline.
    """
    outgoing_edges = 1

    def run(self, root_node: str):  # type: ignore
        return {}, "output_1"


class SklearnQueryClassifier(BaseComponent):
    """
    A node to classify an incoming query into one of two categories using a lightweight sklearn model. Depending on the result, the query flows to a different branch in your pipeline
    and the further processing can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2` from this node.

    Example:
     ```python
        |{
        |pipe = Pipeline()
        |pipe.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
        |pipe.add_node(component=elastic_retriever, name="ElasticRetriever", inputs=["QueryClassifier.output_2"])
        |pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

        |# Keyword queries will use the ElasticRetriever
        |pipe.run("kubernetes aws")

        |# Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
        |pipe.run("How to manage kubernetes on aws")

     ```

    Models:

    Pass your own `Sklearn` binary classification model or use one of the following pretrained ones:
    1) Keywords vs. Questions/Statements (Default)
       query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle)
       query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle)
       output_1 => question/statement
       output_2 => keyword query
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


    2) Questions vs. Statements
       query_classifier can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/model.pickle)
       query_vectorizer can be found [here](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/vectorizer.pickle)
       output_1 => question
       output_2 => statement
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)

    See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.

    """

    outgoing_edges = 2

    def __init__(
        self,
        model_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/model.pickle",
        vectorizer_name_or_path: Union[
            str, Any
        ] = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/vectorizer.pickle"
    ):
        """
        :param model_name_or_path: Gradient boosting based binary classifier to classify between keyword vs statement/question
        queries or statement vs question queries.
        :param vectorizer_name_or_path: A ngram based Tfidf vectorizer for extracting features from query.
        """
        if (
            (not isinstance(model_name_or_path, Path))
            and (not isinstance(model_name_or_path, str))
        ) or (
            (not isinstance(vectorizer_name_or_path, Path))
            and (not isinstance(vectorizer_name_or_path, str))
        ):
            raise TypeError(
                "model_name_or_path and vectorizer_name_or_path must either be of type Path or str"
            )

        # save init parameters to enable export of component config as YAML
        self.set_config(model_name_or_path=model_name_or_path, vectorizer_name_or_path=vectorizer_name_or_path)

        if isinstance(model_name_or_path, Path):
            file_url = urllib.request.pathname2url(r"{}".format(model_name_or_path))
            model_name_or_path = f"file:{file_url}"

        if isinstance(vectorizer_name_or_path, Path):
            file_url = urllib.request.pathname2url(r"{}".format(vectorizer_name_or_path))
            vectorizer_name_or_path = f"file:{file_url}"

        self.model = pickle.load(urllib.request.urlopen(model_name_or_path))
        self.vectorizer = pickle.load(urllib.request.urlopen(vectorizer_name_or_path))


    def run(self, query):
        query_vector = self.vectorizer.transform([query])

        is_question: bool = self.model.predict(query_vector)[0]
        if is_question:
            return {}, "output_1"
        else:
            return {}, "output_2"


class TransformersQueryClassifier(BaseComponent):
    """
    A node to classify an incoming query into one of two categories using a (small) BERT transformer model. Depending on the result, the query flows to a different branch in your pipeline
    and the further processing can be customized. You can define this by connecting the further pipeline to either `output_1` or `output_2` from this node.

    Example:
     ```python
        |{
        |pipe = Pipeline()
        |pipe.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
        |pipe.add_node(component=elastic_retriever, name="ElasticRetriever", inputs=["QueryClassifier.output_2"])
        |pipe.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])

        |# Keyword queries will use the ElasticRetriever
        |pipe.run("kubernetes aws")

        |# Semantic queries (questions, statements, sentences ...) will leverage the DPR retriever
        |pipe.run("How to manage kubernetes on aws")

     ```

    Models:

    Pass your own `Transformer` binary classification model from file/huggingface or use one of the following pretrained ones hosted on Huggingface:
    1) Keywords vs. Questions/Statements (Default)
       model_name_or_path="shahrukhx01/bert-mini-finetune-question-detection"
       output_1 => question/statement
       output_2 => keyword query
       [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier/readme.txt)


    2) Questions vs. Statements
    `model_name_or_path`="shahrukhx01/question-vs-statement-classifier"
     output_1 => question
     output_2 => statement
     [Readme](https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/readme.txt)

     See also the [tutorial](https://haystack.deepset.ai/tutorials/pipelines) on pipelines.
    """

    outgoing_edges = 2

    def __init__(
        self,
        model_name_or_path: Union[
            Path, str
        ] = "shahrukhx01/bert-mini-finetune-question-detection"
    ):
        """
        :param model_name_or_path: Transformer based fine tuned mini bert model for query classification
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(model_name_or_path=model_name_or_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.query_classification_pipeline = TextClassificationPipeline(
            model=model, tokenizer=tokenizer
        )

    def run(self, query):

        is_question: bool = (
            self.query_classification_pipeline(query)[0]["label"] == "LABEL_1"
        )

        if is_question:
            return {}, "output_1"
        else:
            return {}, "output_2"


class JoinDocuments(BaseComponent):
    """
    A node to join documents outputted by multiple retriever nodes.

    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    """

    outgoing_edges = 1

    def __init__(
        self, join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None
    ):
        """
        :param join_mode: `concatenate` to combine documents from multiple retrievers or `merge` to aggregate scores of
                          individual documents.
        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                        to each retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k_join: Limit documents to top_k based on the resulting scores of the join.
        """
        assert join_mode in ["concatenate", "merge"], f"JoinDocuments node does not support '{join_mode}' join_mode."

        assert not (
            weights is not None and join_mode == "concatenate"
        ), "Weights are not compatible with 'concatenate' join_mode."

        # save init parameters to enable export of component config as YAML
        self.set_config(join_mode=join_mode, weights=weights, top_k_join=top_k_join)

        self.join_mode = join_mode
        self.weights = [float(i)/sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join

    def run(self, inputs: List[dict]):  # type: ignore
        if self.join_mode == "concatenate":
            document_map = {}
            for input_from_node in inputs:
                for doc in input_from_node["documents"]:
                    document_map[doc.id] = doc
        elif self.join_mode == "merge":
            document_map = {}
            if self.weights:
                weights = self.weights
            else:
                weights = [1/len(inputs)] * len(inputs)
            for input_from_node, weight in zip(inputs, weights):
                for doc in input_from_node["documents"]:
                    if document_map.get(doc.id):  # document already exists; update score
                        document_map[doc.id].score += doc.score * weight
                    else:  # add the document in map
                        document_map[doc.id] = deepcopy(doc)
                        document_map[doc.id].score *= weight
        else:
            raise Exception(f"Invalid join_mode: {self.join_mode}")

        documents = sorted(document_map.values(), key=lambda d: d.score, reverse=True)

        if self.top_k_join:
            documents = documents[: self.top_k_join]
        output = {"documents": documents, "labels": inputs[0].get("labels", None)}
        return output, "output_1"


class RayPipeline(Pipeline):
    """
    Ray (https://ray.io) is a framework for distributed computing.

    Ray allows distributing a Pipeline's components across a cluster of machines. The individual components of a
    Pipeline can be independently scaled. For instance, an extractive QA Pipeline deployment can have three replicas
    of the Reader and a single replica for the Retriever. It enables efficient resource utilization by horizontally
    scaling Components.

    To set the number of replicas, add  `replicas` in the YAML config for the node in a pipeline:

            ```yaml
            |    components:
            |        ...
            |
            |    pipelines:
            |        - name: ray_query_pipeline
            |          type: RayPipeline
            |          nodes:
            |            - name: ESRetriever
            |              replicas: 2  # number of replicas to create on the Ray cluster
            |              inputs: [ Query ]
            ```

    A RayPipeline can only be created with a YAML Pipeline config.
    >>> from haystack.pipeline import RayPipeline
    >>> pipeline = RayPipeline.load_from_yaml(path="my_pipelines.yaml", pipeline_name="my_query_pipeline")
    >>> pipeline.run(query="What is the capital of Germany?")

    By default, RayPipelines creates an instance of RayServe locally. To connect to an existing Ray instance,
    set the `address` parameter when creating the RayPipeline instance.
    """
    def __init__(self, address: str = None, **kwargs):
        """
        :param address: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
        :param kwargs: Optional parameters for initializing Ray.
        """
        ray.init(address=address, **kwargs)
        serve.start()
        super().__init__()

    @classmethod
    def load_from_yaml(
            cls,
            path: Path, pipeline_name: Optional[str] = None,
            overwrite_with_env_variables: bool = True,
            address: Optional[str] = None,
            **kwargs,
    ):
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
        :param address: The IP address for the Ray cluster. If set to None, a local Ray instance is started.
        """
        data, pipeline_config, definitions = cls._read_yaml(
            path=path, pipeline_name=pipeline_name, overwrite_with_env_variables=overwrite_with_env_variables
        )
        pipeline = cls(address=address, **kwargs)

        for node_config in pipeline_config["nodes"]:
            if pipeline.root_node is None:
                root_node = node_config["inputs"][0]
                if root_node in ["Query", "File"]:
                    pipeline.root_node = root_node
                    handle = cls._create_ray_deployment(component_name=root_node, pipeline_config=data)
                    pipeline._add_ray_deployment_in_graph(handle=handle, name=root_node, outgoing_edges=1,  inputs=[])
                else:
                    raise KeyError(f"Root node '{root_node}' is invalid. Available options are 'Query' and 'File'.")

            name = node_config["name"]
            component_type = definitions[name]["type"]
            component_class = BaseComponent.get_subclass(component_type)
            replicas = next(node for node in pipeline_config["nodes"] if node["name"] == name).get("replicas", 1)
            handle = cls._create_ray_deployment(component_name=name, pipeline_config=data, replicas=replicas)
            pipeline._add_ray_deployment_in_graph(
                handle=handle,
                name=name,
                outgoing_edges=component_class.outgoing_edges,
                inputs=node_config.get("inputs", []),
            )

        return pipeline

    @classmethod
    def _create_ray_deployment(cls, component_name: str, pipeline_config: dict, replicas: int = 1):
        """
        Create a Ray Deployment for the Component.

        :param component_name: Class name of the Haystack Component.
        :param pipeline_config: The Pipeline config YAML parsed as a dict.
        :param replicas: By default, a single replica of the component is created. It can be
                         configured by setting `replicas` parameter in the Pipeline YAML.
        """
        RayDeployment = serve.deployment(_RayDeploymentWrapper, name=component_name, num_replicas=replicas)  # type: ignore
        RayDeployment.deploy(pipeline_config, component_name)
        handle = RayDeployment.get_handle()
        return handle

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
    ):
        has_next_node = True
        current_node_id = self.root_node
        input_dict = {"root_node": self.root_node, "params": params}
        if query:
            input_dict["query"] = query
        if file_paths:
            input_dict["file_paths"] = file_paths
        if labels:
            input_dict["labels"] = labels
        if documents:
            input_dict["documents"] = documents
        if meta:
            input_dict["meta"] = meta

        output_dict = None

        while has_next_node:
            output_dict, stream_id = ray.get(self.graph.nodes[current_node_id]["component"].remote(**input_dict))
            input_dict = output_dict
            next_nodes = self.get_next_nodes(current_node_id, stream_id)

            if len(next_nodes) > 1:
                join_node_id = list(nx.neighbors(self.graph, next_nodes[0]))[0]
                if set(self.graph.predecessors(join_node_id)) != set(next_nodes):
                    raise NotImplementedError(
                        "The current pipeline does not support multiple levels of parallel nodes."
                    )
                inputs_for_join_node: dict = {"inputs": []}
                for n_id in next_nodes:
                    output = self.graph.nodes[n_id]["component"].run(**input_dict)
                    inputs_for_join_node["inputs"].append(output)
                input_dict = inputs_for_join_node
                current_node_id = join_node_id
            elif len(next_nodes) == 1:
                current_node_id = next_nodes[0]
            else:
                has_next_node = False

        return output_dict

    def add_node(self, component, name: str, inputs: List[str]):
        raise NotImplementedError(
            "The current implementation of RayPipeline only supports loading Pipelines from a YAML file."
        )

    def _add_ray_deployment_in_graph(self, handle, name: str, outgoing_edges: int, inputs: List[str]):
        """
        Add the Ray deployment handle in the Pipeline Graph.

        :param handle: Ray deployment `handle` to add in the Pipeline Graph. The handle allow calling a Ray deployment
                       from Python: https://docs.ray.io/en/master/serve/package-ref.html#servehandle-api.
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
            self.node = BaseComponent.load_from_pipeline_config(pipeline_config, component_name)

    def __call__(self, *args, **kwargs):
        """
        Ray calls this method which is then re-directed to the corresponding component's run().
        """
        return self.node._dispatch_run(*args, **kwargs)


class Docs2Answers(BaseComponent):
    outgoing_edges = 1

    def __init__(self):
        self.set_config()

    def run(self, query, documents):  # type: ignore
        # conversion from Document -> Answer
        answers = []
        for doc in documents:
            # For FAQ style QA use cases
            if "answer" in doc.meta:
                cur_answer = {
                    "query": doc.text,
                    "answer": doc.meta["answer"],
                    "document_id": doc.id,
                    "context": doc.meta["answer"],
                    "score": doc.score,
                    "offset_start": 0,
                    "offset_end": len(doc.meta["answer"]),
                    "meta": doc.meta,
                }
            else:
                # Regular docs
                cur_answer = {
                    "query": None,
                    "answer": None,
                    "document_id": doc.id,
                    "context": doc.text,
                    "score": doc.score,
                    "offset_start": None,
                    "offset_end": None,
                    "meta": doc.meta,
                }
            answers.append(cur_answer)

        output = {"query": query, "answers": answers}

        return output, "output_1"

class MostSimilarDocumentsPipeline(BaseStandardPipeline):
    def __init__(self, document_store: BaseDocumentStore):
        """
        Initialize a Pipeline for finding the most similar documents to a given document.
        This pipeline can be helpful if you already show a relevant document to your end users and they want to search for just similar ones.  

        :param document_store: Document Store instance with already stored embeddings. 
        """
        self.document_store = document_store

    def run(self, document_ids: List[str], top_k: int = 5):
        """
        :param document_ids: document ids
        :param top_k: How many documents id to return against single document
        """
        similar_documents: list = []
        self.document_store.return_embedding = True  # type: ignore

        for document in self.document_store.get_documents_by_id(ids=document_ids):
            similar_documents.append(self.document_store.query_by_embedding(query_emb=document.embedding,
                                                                            return_embedding=False,
                                                                            top_k=top_k))

        self.document_store.return_embedding = False  # type: ignore
        return similar_documents

