from typing import Dict, List, Optional, Any, Union

import copy
import inspect
import logging
import os
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from pandas.core.frame import DataFrame
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph
from haystack.nodes.evaluator.evaluator import calculate_em_str_multi, calculate_f1_str_multi, semantic_answer_similarity

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

from haystack.schema import EvaluationResult, MultiLabel, Document
from haystack.nodes.base import BaseComponent
from haystack.document_stores.base import BaseDocumentStore


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
        debug: Optional[bool] = None
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
                          they received and the output they generated. All debug information can 
                          then be found in the dict returned by this method under the key "_debug"
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

    def eval(
        self,
        labels: List[MultiLabel],
        params: Optional[dict] = None,
        sas_model_name_or_path: str = None
    ) -> EvaluationResult:
        """
            Evaluates the pipeline by running the pipeline once per query in debug mode 
            and putting together all data that is needed for evaluation, e.g. calculating metrics.

            :param labels: The labels to evaluate on
            :param params: Dictionary of parameters to be dispatched to the nodes. 
                        If you want to pass a param to all nodes, you can just use: {"top_k":10}
                        If you want to pass it to targeted nodes, you can do:
                        {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
            :param sas_model_name_or_path: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
                        The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
                        Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture.
                        More info in the paper: https://arxiv.org/abs/2108.06130
                        Models:
                        - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                        Not all cross encoders can be used because of different return types.
                        If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
                        - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                        - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
                        - Large model for German only: "deepset/gbert-large-sts"
        """    
        eval_result = EvaluationResult()
        queries = [label.query for label in labels]
        for query, label in zip(queries, labels):
            predictions = self.run(query=query, labels=label, params=params, debug=True)
            
            for node_name in predictions["_debug"].keys():
                node_output = predictions["_debug"][node_name]["output"]
                df = self._build_eval_dataframe(
                    query, label, node_name, node_output)
                eval_result.append(node_name, df)
        
        # add sas values in batch mode for whole Dataframe
        # this is way faster than if we calculate it for each query separately
        if sas_model_name_or_path is not None:
            for df in eval_result.node_results.values():
                if len(df[df["type"] == "answer"]) > 0:
                    gold_labels = df["gold_answers"].values
                    predictions = [[a] for a in df["answer"].values]
                    sas, _ = semantic_answer_similarity(predictions=predictions, gold_labels=gold_labels, 
                                                sas_model_name_or_path=sas_model_name_or_path)
                    df["sas"] = sas

        return eval_result

    def _build_eval_dataframe(self, 
        query: str, 
        query_labels: MultiLabel, 
        node_name: str, 
        node_output: dict
    ) -> DataFrame:
        """
        Builds a Dataframe for each query from which evaluation metrics can be calculated.
        Currently only answer or document returning nodes are supported, returns None otherwise.
        
        Each row contains either an answer or a document that has been retrieved during evaluation.
        Rows are being enriched with basic infos like rank, query, type or node.
        Additional answer or document specific evaluation infos like gold labels 
        and metrics depicting whether the row matches the gold labels are included, too.
        """
        df: DataFrame = pd.DataFrame()

        if query_labels is None or query_labels.labels is None:
            logger.warning(f"There is no label for query '{query}'. Query will be omitted.")
            return df

        # remarks for no_answers:
        # Single 'no_answer'-labels are not contained in MultiLabel aggregates.
        # If all labels are no_answers, MultiLabel.answers will be [""] and the other aggregates []
        gold_answers = query_labels.answers
        gold_offsets_in_documents = query_labels.gold_offsets_in_documents
        gold_document_ids = query_labels.document_ids
        gold_document_contents = query_labels.document_contents

        # if node returned answers, include answer specific info:
        # - the answer returned itself
        # - the document_id the answer was found in
        # - the position or offsets within the document the answer was found
        # - the surrounding context of the answer within the document
        # - the gold answers
        # - the positon or offsets of the gold answer within the document
        # - the gold document ids containing the answer
        # - the exact_match metric depicting if the answer exactly matches the gold label
        # - the f1 metric depicting how well the answer overlaps with the gold label on token basis
        # - the sas metric depciting how well the answer matches the gold label on a semantic basis.
        #   this will be calculated on all queries in eval() for performance reasons if a sas model has been provided
        answers = node_output.get("answers", None)
        if answers is not None:
            answer_cols_to_keep = ["answer", "document_id", "offsets_in_document", "context"]
            df_answers = pd.DataFrame(answers, columns=answer_cols_to_keep)
            if len(df_answers) > 0:
                df_answers["type"] = "answer"
                df_answers["gold_answers"] = [gold_answers] * len(df_answers)
                df_answers["gold_offsets_in_documents"] = [gold_offsets_in_documents] * len(df_answers)
                df_answers["gold_document_ids"] = [gold_document_ids] * len(df_answers)
                df_answers["exact_match"] = df_answers.apply(
                    lambda row: calculate_em_str_multi(gold_answers, row["answer"]), axis=1)
                df_answers["f1"] = df_answers.apply(
                    lambda row: calculate_f1_str_multi(gold_answers, row["answer"]), axis=1)
                df_answers["rank"] = np.arange(1, len(df_answers)+1)
                df = pd.concat([df, df_answers])

        # if node returned documents, include document specific info:
        # - the document_id
        # - the content of the document
        # - the gold document ids
        # - the gold document contents
        # - the gold_id_match metric depicting whether one of the gold document ids matches the document
        # - the answer_match metric depicting whether the document contains the answer
        # - the gold_id_or_answer_match metric depicting whether one of the former two conditions are met
        documents = node_output.get("documents", None)
        if documents is not None:
            document_cols_to_keep = ["content", "id"]
            df_docs = pd.DataFrame(documents, columns=document_cols_to_keep)
            if len(df_docs) > 0:
                df_docs = df_docs.rename(columns={"id": "document_id"})
                df_docs["type"] = "document"
                df_docs["gold_document_ids"] = [gold_document_ids] * len(df_docs)
                df_docs["gold_document_contents"] = [gold_document_contents] * len(df_docs)
                df_docs["gold_id_match"] = df_docs.apply(
                    lambda row: 1.0 if row["document_id"] in gold_document_ids else 0.0, axis=1)
                df_docs["answer_match"] = df_docs.apply(
                    lambda row: 
                        1.0 if not query_labels.no_answer 
                            and any(gold_answer in row["content"] for gold_answer in gold_answers) 
                        else 0.0, 
                    axis=1)
                df_docs["gold_id_or_answer_match"] = df_docs.apply(
                    lambda row: max(row["gold_id_match"], row["answer_match"]), axis=1)
                df_docs["rank"] = np.arange(1, len(df_docs)+1)
                df = pd.concat([df, df_docs])

        # add general info
        df["node"] = node_name
        df["query"] = query

        return df

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
        | from haystack.document_stores.base import BaseDocumentStore
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

    
    def _format_document_answer(self, document_or_answer: dict):
        return "\n \t".join([f'{name}: {value}' for name, value in document_or_answer.items()])

    def _format_wrong_sample(self, query: dict):
        metrics = "\n \t".join([f'{name}: {value}' for name, value in query['metrics'].items()])
        documents = "\n\n \t".join([self._format_document_answer(doc) for doc in query.get('documents', [])])
        documents = f"Documents: \n \t{documents}\n" if len(documents) > 0 else ""
        answers = "\n\n \t".join([self._format_document_answer(answer) for answer in query.get('answers', [])])
        answers = f"Answers: \n \t{answers}\n" if len(answers) > 0 else ""
        gold_document_ids = "\n \t".join(query['gold_document_ids'])
        gold_answers = "\n \t".join(query.get('gold_answers', []))
        gold_answers = f"Gold Answers: \n \t{gold_answers}\n" if len(gold_answers) > 0 else ""
        s = (
            f"Query: \n \t{query['query']}\n"
            f"{gold_answers}"
            f"Gold Document Ids: \n \t{gold_document_ids}\n"
            f"Metrics: \n \t{metrics}\n"
            f"{answers}"
            f"{documents}"
            f"_______________________________________________________"
        )
        return s

    def _format_wrong_samples_node(self, node_name: str, wrong_samples_formatted: str):
        s = (
            f"                Wrong {node_name} Examples\n"
            f"=======================================================\n"
            f"{wrong_samples_formatted}\n"
            f"=======================================================\n"
        )
        return s

    def _format_wrong_samples_report(self, eval_result: EvaluationResult, n_wrong_examples: int = 3):
        examples = {
            node: eval_result.wrong_examples(node, doc_relevance_col="gold_id_or_answer_match", n=n_wrong_examples) 
                for node in eval_result.node_results.keys()
        }
        examples_formatted = {
            node: "\n".join([self._format_wrong_sample(example) for example in examples]) 
                for node, examples in examples.items()
        }

        return "\n".join([self._format_wrong_samples_node(node, examples) for node, examples in examples_formatted.items()])

    def _format_pipeline_node(self, node: str, metrics: dict, metrics_top_1):
        metrics = metrics.get(node, {})
        metrics_top_1 = {f"{metric}_top_1": value for metric, value in metrics_top_1.get(node, {}).items()}
        node_metrics = {**metrics, **metrics_top_1}
        node_metrics_formatted = "\n".join(sorted([f"                        | {metric}: {value:5.3}" for metric, value in node_metrics.items()])) 
        node_metrics_formatted = f"{node_metrics_formatted}\n" if len(node_metrics_formatted) > 0 else ""
        s = (
            f"                      {node}\n"
            f"                        |\n"
            f"{node_metrics_formatted}"
            f"                        |"
        )
        return s

    def _format_pipeline_overview(self, metrics: dict, metrics_top_1: dict):
        pipeline_overview = "\n".join([self._format_pipeline_node(node, metrics, metrics_top_1) for node in self.graph.nodes])
        s = (
            f"================== Evaluation Report ==================\n"
            f"=======================================================\n"
            f"                   Pipeline Overview\n"
            f"=======================================================\n"
            f"{pipeline_overview}\n"
            f"                      Output\n"
            f"=======================================================\n"
        )
        return s

    def print_eval_report(
        self, 
        eval_result: EvaluationResult, 
        n_wrong_examples: int = 3, 
        metrics_filter: Optional[Dict[str, List[str]]] = None):
        """
        Prints evaluation report containing a metrics funnel and worst queries for further analysis.
        
        :param eval_result: The evaluation result, can be obtained by running eval().
        :param n_wrong_examples: The number of worst queries to show.
        :param metrics_filter: The metrics to show per node. If None all metrics will be shown.
        """
        if any(degree > 1 for node, degree in self.graph.out_degree):
            logger.warning("Pipelines with junctions are currently not supported.")
            return
        
        metrics_top_n = eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match")
        metrics_top_1 = eval_result.calculate_metrics(doc_relevance_col="gold_id_or_answer_match", simulated_top_k_reader=1)
        if metrics_filter is not None:
            metrics_top_n = {node: metrics if node not in metrics_filter 
                                    else {metric: value for metric, value in metrics.items() if metric in metrics_filter[node]} 
                                    for node, metrics in metrics_top_n.items()}
            metrics_top_1 = {node: metrics if node not in metrics_filter 
                                    else {metric: value for metric, value in metrics.items() if metric in metrics_filter[node]} 
                                    for node, metrics in metrics_top_1.items()}
        pipeline_overview = self._format_pipeline_overview(metrics_top_n, metrics_top_1)        
        wrong_samples_report = self._format_wrong_samples_report(eval_result=eval_result, n_wrong_examples=n_wrong_examples)

        print(
            f"{pipeline_overview}\n"
            f"{wrong_samples_report}")


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
