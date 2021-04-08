import logging
import os
import traceback
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict

import networkx as nx
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack import BaseComponent
from haystack.generator.base import BaseGenerator
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.summarizer.base import BaseSummarizer
from haystack.translator.base import BaseTranslator
from haystack.knowledge_graph.base import BaseKnowledgeGraph
from haystack.graph_retriever.base import BaseGraphRetriever


logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

    Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
    flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
    Reader from multiple Retrievers, or re-ranking of candidate documents.
    """

    def __init__(self, pipeline_type: str = "Query"):
        self.graph = DiGraph()
        if pipeline_type == "Query":
            self.root_node_id = "Query"
            self.graph.add_node("Query", component=RootNode())
        elif pipeline_type == "Indexing":
            self.root_node_id = "File"
            self.graph.add_node("File", component=RootNode())
        else:
            raise Exception(f"pipeline_type '{pipeline_type}' is not valid. Supported types are 'Query' & 'Indexing'.")

        self.pipeline_type = pipeline_type
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
        self.graph.add_node(name, component=component, inputs=inputs)

        if len(self.graph.nodes) == 2:  # first node added; connect with Root
            self.graph.add_edge(self.root_node_id, name, label="output_1")
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

    def run(self, **kwargs):
        node_output = None
        queue = {
            self.root_node_id: {"pipeline_type": self.pipeline_type, **kwargs}
        }  # ordered dict with "node_id" -> "input" mapping that acts as a FIFO queue
        i = 0  # the first item is popped off the queue unless it is a "join" node with unprocessed predecessors
        while queue:
            node_id = list(queue.keys())[i]
            node_input = queue[node_id]
            predecessors = set(nx.ancestors(self.graph, node_id))
            if predecessors.isdisjoint(set(queue.keys())):  # only execute if predecessor nodes are executed
                try:
                    logger.debug(f"Running node `{node_id}` with input `{node_input}`")
                    node_output, stream_id = self.graph.nodes[node_id]["component"].run(**node_input)
                except Exception as e:
                    tb = traceback.format_exc()
                    raise Exception(f"Exception while running node `{node_id}` with input `{node_input}`: {e}, full stack trace: {tb}")
                queue.pop(node_id)
                next_nodes = self.get_next_nodes(node_id, stream_id)
                for n in next_nodes:  # add successor nodes with corresponding inputs to the queue
                    if queue.get(n):  # concatenate inputs if it's a join node
                        existing_input = queue[n]
                        if "inputs" not in existing_input.keys():
                            updated_input = {"inputs": [existing_input, node_output]}
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
            |   version: '0.7'
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

        definitions = {}  # definitions of each component from the YAML.
        for definition in data["components"]:
            if overwrite_with_env_variables:
                cls._overwrite_with_env_variables(definition)
            name = definition.pop("name")
            definitions[name] = definition

        pipeline = cls(pipeline_type=pipeline_config["type"])

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

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: int = 10, top_k_reader: int = 10):
        output = self.pipeline.run(
            query=query, filters=filters, top_k_retriever=top_k_retriever, top_k_reader=top_k_reader
        )
        return output


class DocumentSearchPipeline(BaseStandardPipeline):
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize a Pipeline for semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: Optional[int] = None):
        output = self.pipeline.run(query=query, filters=filters, top_k_retriever=top_k_retriever)
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

    def run(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k_retriever: Optional[int] = None,
        top_k_generator: Optional[int] = None
    ):
        output = self.pipeline.run(
            query=query, filters=filters, top_k_retriever=top_k_retriever, top_k_generator=top_k_generator
        )
        return output


class SearchSummarizationPipeline(BaseStandardPipeline):
    def __init__(self, summarizer: BaseSummarizer, retriever: BaseRetriever):
        """
        Initialize a Pipeline that retrieves documents for a query and then summarizes those documents.

        :param summarizer: Summarizer instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=summarizer, name="Summarizer", inputs=["Retriever"])

    def run(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k_retriever: Optional[int] = None,
        generate_single_summary: Optional[bool] = None,
        return_in_answer_format: bool = False,
    ):
        """
        :param query: Your search query
        :param filters:
        :param top_k_retriever: Number of top docs the retriever should pass to the summarizer.
                                The higher this value, the slower your pipeline.
        :param generate_single_summary: Whether to generate single summary from all retrieved docs (True) or one per doc (False).
        :param return_in_answer_format: Whether the results should be returned as documents (False) or in the answer format used in other QA pipelines (True).
                                        With the latter, you can use this pipeline as a "drop-in replacement" for other QA pipelines.
        """
        output = self.pipeline.run(
            query=query, filters=filters, top_k_retriever=top_k_retriever, generate_single_summary=generate_single_summary
        )

        # Convert to answer format to allow "drop-in replacement" for other QA pipelines
        if return_in_answer_format:
            results: Dict = {"query": query, "answers": []}
            docs = deepcopy(output["documents"])
            for doc in docs:
                cur_answer = {
                    "query": query,
                    "answer": doc.text,
                    "document_id": doc.id,
                    "context": doc.meta.pop("context"),
                    "score": None,
                    "probability": None,
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

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: Optional[int] = None):
        output = self.pipeline.run(query=query, filters=filters, top_k_retriever=top_k_retriever)
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
                "probability": doc.probability,
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


class RootNode:
    outgoing_edges = 1

    def run(self, **kwargs):
        return kwargs, "output_1"


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
        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k_join

    def run(self, **kwargs):
        inputs = kwargs["inputs"]

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
        if self.top_k:
            documents = documents[: self.top_k]
        output = {"query": inputs[0]["query"], "documents": documents, "labels": inputs[0].get("labels", None)}
        return output, "output_1"
