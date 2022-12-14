# pylint: disable=too-many-public-methods

from __future__ import annotations

import datetime
import itertools
from datetime import timedelta
from functools import partial
from hashlib import sha1
from typing import Dict, List, Optional, Any, Set, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import copy
import json
import inspect
import logging
import tempfile
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import networkx as nx
from pandas.core.frame import DataFrame
from tqdm.auto import tqdm
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack import __version__
from haystack.nodes.evaluator.evaluator import semantic_answer_similarity
from haystack.modeling.evaluation.squad import compute_f1 as calculate_f1_str
from haystack.modeling.evaluation.squad import compute_exact as calculate_em_str
from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
    validate_config,
    _add_node_to_pipeline_graph,
    _init_pipeline_graph,
    VALID_ROOT_NODES,
)
from haystack.pipelines.utils import generate_code, print_eval_report
from haystack.utils import DeepsetCloud, calculate_context_similarity
from haystack.schema import Answer, EvaluationResult, MultiLabel, Document, Span
from haystack.errors import HaystackError, PipelineError, PipelineConfigError
from haystack.nodes import BaseGenerator, Docs2Answers, BaseReader, BaseSummarizer, BaseTranslator, QuestionGenerator
from haystack.nodes.base import BaseComponent, RootNode
from haystack.nodes.retriever.base import BaseRetriever
from haystack.document_stores.base import BaseDocumentStore
from haystack.telemetry import send_event, send_custom_event
from haystack.utils.experiment_tracking import MLflowTrackingHead, Tracker as tracker


logger = logging.getLogger(__name__)


ROOT_NODE_TO_PIPELINE_NAME = {"query": "query", "file": "indexing"}
CODE_GEN_DEFAULT_COMMENT = "This code has been generated."
TRACKING_TOOL_TO_HEAD = {"mlflow": MLflowTrackingHead}


class Pipeline:
    """
    Pipeline brings together building blocks to build a complex search pipeline with Haystack and user-defined components.

    Under the hood, a Pipeline is represented as a directed acyclic graph of component nodes. You can use it for custom query flows with the option to branch queries (for example, extractive question answering and keyword match query), merge candidate documents for a Reader from multiple Retrievers, or re-ranking of candidate documents.
    """

    def __init__(self):
        self.graph = DiGraph()
        self.init_time = datetime.datetime.now(datetime.timezone.utc)
        self.time_of_last_sent_event = datetime.datetime.now(datetime.timezone.utc)
        self.event_time_interval = datetime.timedelta(hours=24)
        self.event_run_total_threshold = 100
        self.last_window_run_total = 0
        self.run_total = 0
        self.sent_event_in_window = False

    @property
    def root_node(self) -> Optional[str]:
        """
        Returns the root node of the pipeline's graph.
        """
        if len(self.graph.nodes) < 1:
            return None
        return list(self.graph.nodes)[0]  # List conversion is required, see networkx docs

    @property
    def components(self) -> Dict[str, BaseComponent]:
        """
        Returns all components used by this pipeline.
        Note that this also includes such components that are being utilized by other components only and are not being used as a pipeline node directly.
        """
        all_components = self._find_all_components()
        return {component.name: component for component in all_components if component.name is not None}

    def _find_all_components(self, seed_components: Optional[List[BaseComponent]] = None) -> Set[BaseComponent]:
        """
        Finds all components given the provided seed components.
        Components are found by traversing the provided seed components and their utilized components.
        If seed_components is None, the node components (except the root node) of the pipeline will be used as seed components.
        """
        if seed_components is None:
            seed_components = [
                attributes["component"]
                for attributes in self.graph.nodes.values()
                if not isinstance(attributes["component"], RootNode)
            ]

        distinct_components = set(seed_components)
        for component in seed_components:
            sub_components = self._find_all_components(component.utilized_components)
            distinct_components.update(sub_components)
        return distinct_components

    def to_code(
        self, pipeline_variable_name: str = "pipeline", generate_imports: bool = True, add_comment: bool = False
    ) -> str:
        """
        Returns the code to create this pipeline as string.

        :param pipeline_variable_name: The variable name of the generated pipeline.
                                       Default value is 'pipeline'.
        :param generate_imports: Whether to include the required import statements into the code.
                                 Default value is True.
        :param add_comment: Whether to add a preceding comment that this code has been generated.
                            Default value is False.
        """
        pipeline_config = self.get_config()
        code = generate_code(
            pipeline_config=pipeline_config,
            pipeline_variable_name=pipeline_variable_name,
            generate_imports=generate_imports,
            comment=CODE_GEN_DEFAULT_COMMENT if add_comment else None,
        )
        return code

    def to_notebook_cell(
        self, pipeline_variable_name: str = "pipeline", generate_imports: bool = True, add_comment: bool = True
    ):
        """
        Creates a new notebook cell with the code to create this pipeline.

        :param pipeline_variable_name: The variable name of the generated pipeline.
                                       Default value is 'pipeline'.
        :param generate_imports: Whether to include the required import statements into the code.
                                 Default value is True.
        :param add_comment: Whether to add a preceding comment that this code has been generated.
                            Default value is True.
        """
        pipeline_config = self.get_config()
        code = generate_code(
            pipeline_config=pipeline_config,
            pipeline_variable_name=pipeline_variable_name,
            generate_imports=generate_imports,
            comment=CODE_GEN_DEFAULT_COMMENT if add_comment else None,
            add_pipeline_cls_import=False,
        )
        try:
            get_ipython().set_next_input(code)  # type: ignore
        except NameError:
            logger.error("Could not create notebook cell. Make sure you're running in a notebook environment.")

    @classmethod
    def load_from_deepset_cloud(
        cls,
        pipeline_config_name: str,
        pipeline_name: str = "query",
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        overwrite_with_env_variables: bool = False,
    ):
        """
        Load Pipeline from Deepset Cloud defining the individual components and how they're tied together to form
        a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        In order to get a list of all available pipeline_config_names, call `list_pipelines_on_deepset_cloud()`.
        Use the returned `name` as `pipeline_config_name`.

        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
                                     To get a list of all available pipeline_config_names, call `list_pipelines_on_deepset_cloud()`.
        :param pipeline_name: specifies which pipeline to load from config.
                              Deepset Cloud typically provides a 'query' and a 'index' pipeline per config.
        :param workspace: workspace in Deepset Cloud
        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param overwrite_with_env_variables: Overwrite the config with environment variables. For example,
                                             to change return_no_answer param for a FARMReader, an env
                                             variable 'READER_PARAMS_RETURN_NO_ANSWER=False' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        client = DeepsetCloud.get_pipeline_client(
            api_key=api_key, api_endpoint=api_endpoint, workspace=workspace, pipeline_config_name=pipeline_config_name
        )
        pipeline_config = client.get_pipeline_config()

        # update document store params in order to connect to correct index
        for component_config in pipeline_config["components"]:
            if component_config["type"] == "DeepsetCloudDocumentStore":
                params = component_config.get("params", {})
                params["index"] = pipeline_config_name
                params["workspace"] = workspace
                if api_endpoint is not None:
                    params["api_endpoint"] = api_endpoint
                if api_key is not None:
                    params["api_key"] = api_key
                component_config["params"] = params

        del pipeline_config["name"]  # Would fail validation otherwise
        pipeline = cls.load_from_config(
            pipeline_config=pipeline_config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
        )
        return pipeline

    @classmethod
    def list_pipelines_on_deepset_cloud(
        cls, workspace: str = "default", api_key: Optional[str] = None, api_endpoint: Optional[str] = None
    ) -> List[dict]:
        """
        Lists all pipeline configs available on Deepset Cloud.

        :param workspace: workspace in Deepset Cloud
        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.

        Returns:
            list of dictionaries: List[dict]
            each dictionary: {
                        "name": str -> `pipeline_config_name` to be used in `load_from_deepset_cloud()`,
                        "..." -> additional pipeline meta information
                        }
            example:

            ```python
            [{'name': 'my_super_nice_pipeline_config',
                'pipeline_id': '2184e0c1-c6ec-40a1-9b28-5d2768e5efa2',
                'status': 'DEPLOYED',
                'created_at': '2022-02-01T09:57:03.803991+00:00',
                'deleted': False,
                'is_default': False,
                'indexing': {'status': 'IN_PROGRESS',
                'pending_file_count': 3,
                'total_file_count': 31}}]
            ```
        """
        client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        pipeline_config_infos = list(client.list_pipeline_configs())
        return pipeline_config_infos

    @classmethod
    def save_to_deepset_cloud(
        cls,
        query_pipeline: Pipeline,
        index_pipeline: Pipeline,
        pipeline_config_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Saves a Pipeline config to Deepset Cloud defining the individual components and how they're tied together to form
        a Pipeline. A single config must declare a query pipeline and a index pipeline.

        :param query_pipeline: the query pipeline to save.
        :param index_pipeline: the index pipeline to save.
        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param overwrite: Whether to overwrite the config if it already exists. Otherwise an error is being raised.
        """
        query_config = query_pipeline.get_config()
        index_config = index_pipeline.get_config()
        pipelines = query_config["pipelines"] + index_config["pipelines"]
        all_components = query_config["components"] + index_config["components"]
        distinct_components = [c for c in {component["name"]: component for component in all_components}.values()]
        document_stores = [c for c in distinct_components if c["type"].endswith("DocumentStore")]
        for document_store in document_stores:
            if document_store["type"] != "DeepsetCloudDocumentStore":
                logger.info(
                    f"In order to be used on Deepset Cloud, component '{document_store['name']}' of type '{document_store['type']}' "
                    f"has been automatically converted to type DeepsetCloudDocumentStore. "
                    f"Usually this replacement will result in equivalent pipeline quality. "
                    f"However depending on chosen settings of '{document_store['name']}' differences might occur."
                )
                document_store["type"] = "DeepsetCloudDocumentStore"
                document_store["params"] = {}
        config = {"components": distinct_components, "pipelines": pipelines, "version": __version__}

        client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        pipeline_config_info = client.get_pipeline_config_info(pipeline_config_name=pipeline_config_name)
        if pipeline_config_info:
            if overwrite:
                if pipeline_config_info["status"] == "DEPLOYED":
                    raise ValueError(
                        f"Deployed pipeline configs are not allowed to be updated. Please undeploy pipeline config '{pipeline_config_name}' first."
                    )
                client.update_pipeline_config(config=config, pipeline_config_name=pipeline_config_name)
                logger.info("Pipeline config '%s' successfully updated.", pipeline_config_name)
            else:
                raise ValueError(
                    f"Pipeline config '{pipeline_config_name}' already exists. Set `overwrite=True` to overwrite pipeline config."
                )
        else:
            client.save_pipeline_config(config=config, pipeline_config_name=pipeline_config_name)
            logger.info("Pipeline config '%s' successfully created.", pipeline_config_name)

    @classmethod
    def deploy_on_deepset_cloud(
        cls,
        pipeline_config_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        timeout: int = 60,
        show_curl_message: bool = True,
    ):
        """
        Deploys the pipelines of a pipeline config on Deepset Cloud.
        Blocks until pipelines are successfully deployed, deployment failed or timeout exceeds.
        If pipelines are already deployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        Pipeline config must be present on Deepset Cloud. See save_to_deepset_cloud() for more information.

        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param timeout: The time in seconds to wait until deployment completes.
                        If the timeout is exceeded an error will be raised.
        :param show_curl_message: Whether to print an additional message after successful deployment showing how to query the pipeline using curl.
        """
        client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        client.deploy(pipeline_config_name=pipeline_config_name, timeout=timeout, show_curl_message=show_curl_message)

    @classmethod
    def undeploy_on_deepset_cloud(
        cls,
        pipeline_config_name: str,
        workspace: str = "default",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Undeploys the pipelines of a pipeline config on Deepset Cloud.
        Blocks until pipelines are successfully undeployed, undeployment failed or timeout exceeds.
        If pipelines are already undeployed no action will be taken and an info will be logged.
        If timeout exceeds a TimeoutError will be raised.
        If deployment fails a DeepsetCloudError will be raised.

        Pipeline config must be present on Deepset Cloud. See save_to_deepset_cloud() for more information.

        :param pipeline_config_name: name of the config file inside the Deepset Cloud workspace.
        :param workspace: workspace in Deepset Cloud
        :param api_key: Secret value of the API key.
                        If not specified, will be read from DEEPSET_CLOUD_API_KEY environment variable.
        :param api_endpoint: The URL of the Deepset Cloud API.
                             If not specified, will be read from DEEPSET_CLOUD_API_ENDPOINT environment variable.
        :param timeout: The time in seconds to wait until undeployment completes.
                        If the timeout is exceeded an error will be raised.
        """
        client = DeepsetCloud.get_pipeline_client(api_key=api_key, api_endpoint=api_endpoint, workspace=workspace)
        client.undeploy(pipeline_config_name=pipeline_config_name, timeout=timeout)

    def add_node(self, component: BaseComponent, name: str, inputs: List[str]):
        """
        Add a new node to the pipeline.

        :param component: The object to be called when the data is passed to the node. It can be a Haystack component
                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                          method to process incoming data from predecessor node.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'BM25Retriever' node would always output a single
                       edge with a list of documents. It can be represented as ["BM25Retriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        if len(self.graph.nodes) < 1:
            candidate_roots = [input_node for input_node in inputs if input_node in VALID_ROOT_NODES]
            if len(candidate_roots) != 1:
                raise PipelineConfigError(
                    "The first node of a pipeline must have one single root node "
                    f"as input ({' or '.join(VALID_ROOT_NODES)})."
                )
            self.graph = _init_pipeline_graph(root_node_name=candidate_roots[0])

        # Check for duplicate names before adding the component
        # Note that the very same component must be addable multiple times:
        # E.g. for indexing pipelines it's common to add a retriever first and a document store afterwards.
        # The document store is already being used by the retriever however.
        # Thus the very same document store will be added twice, first as a subcomponent of the retriever and second as a first level node.
        if name in self.components.keys() and self.components[name] != component:
            raise PipelineConfigError(f"A node named '{name}' is already in the pipeline. Choose another name.")

        component_definitions = get_component_definitions(pipeline_config=self.get_config())
        component_definitions[name] = component._component_config

        # Name any nested component before adding them
        component.name = name
        component_names = self._get_all_component_names()
        component_names.add(name)
        self._set_sub_component_names(component, component_names=component_names)

        self.graph = _add_node_to_pipeline_graph(
            graph=self.graph,
            components=component_definitions,
            node={"name": name, "inputs": inputs},
            instance=component,
        )

    def get_node(self, name: str) -> Optional[BaseComponent]:
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        graph_node = self.graph.nodes.get(name)
        component = graph_node["component"] if graph_node else None
        return component

    # FIXME unused and untested. In which cases do we need to set nodes? Can this be removed?
    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.graph.nodes[name]["component"] = component

    def _run_node(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:
        return self.graph.nodes[node_id]["component"]._dispatch_run(**node_input)

    def run(  # type: ignore
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
                    node_output, stream_id = self._run_node(node_id, node_input)
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
        self.send_pipeline_event_if_needed(is_indexing=file_paths is not None)
        return node_output

    def run_batch(  # type: ignore
        self,
        queries: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        """
        Runs the Pipeline in a batch mode, one node at a time. The batch mode means that the Pipeline can take more than one query as input. You can use this method for query pipelines only. When used with an indexing pipeline, it calls the pipeline `run()` method.

        Here's what this method returns for Retriever-Reader pipelines:
        - Single query: Retrieves top-k relevant Documents and returns a list of answers for each retrieved Document.
        - A list of queries: Retrieves top-k relevant Documents for each query and returns a list of answers for each query.

        Here's what this method returns for Reader-only pipelines:
        - Single query + a list of Documents: Applies the query to each Document individually and returns answers    for each single Document.
        - Single query + a list of lists of Documents: Applies the query to each list of Documents and returns aggregated answers for each list of Documents.
        - A list of queries + a list of Documents: Applies each query to each Document individually and returns answers for each query-document pair.
        - A list of queries + a list of lists of Documents: Applies each query to its corresponding Document list and aggregates answers for each list of Documents.


        :param queries: List of search queries (for query pipelines only).
        :param file_paths: The files to index (for indexing pipelines only). If you provide `file_paths` the Pipeline's `run` method instead of `run_batch` is called.
        :param labels: Ground-truth labels that you can use to perform an isolated evaluation of pipelines. These labels are input to nodes in the pipeline.
        :param documents: A list of Document objects or a list of lists of Document objects to be processed by the Pipeline Nodes.
        :param meta: Files' metadata. Used in indexing pipelines in combination with `file_paths`.
        :param params: A dictionary of parameters that you want to pass to the nodes.
                       To pass a parameter to all Nodes, use: `{"top_k": 10}`.
                       To pass a parameter to targeted Nodes, run:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}`
        :param debug: Specifies whether the Pipeline should instruct Nodes to collect debug information
                      about their execution. By default, this information includes the input parameters
                      the Nodes received and the output they generated. You can then find all debug information in the dictionary returned by this method under the key `_debug`.
        """
        if file_paths is not None or meta is not None:
            logger.info(
                "It seems that an indexing Pipeline is run, so using the nodes' run method instead of run_batch."
            )
            if isinstance(queries, list):
                raise PipelineError("For indexing, only a single query can be provided.")
            if isinstance(labels, list):
                raise PipelineError("For indexing, only one MultiLabel object can be provided as labels.")
            flattened_documents: List[Document] = []
            if documents and isinstance(documents[0], list):
                for doc_list in documents:
                    assert isinstance(doc_list, list)
                    flattened_documents.extend(doc_list)
            return self.run(
                query=queries,
                file_paths=file_paths,
                labels=labels,
                documents=flattened_documents,
                meta=meta,
                params=params,
                debug=debug,
            )
        # Validate node names
        self._validate_node_names_in_params(params=params)

        root_node = self.root_node
        if not root_node:
            raise PipelineError("Cannot run a pipeline with no nodes.")

        node_output = None
        queue: Dict[str, Any] = {
            root_node: {"root_node": root_node, "params": params}
        }  # ordered dict with "node_id" -> "input" mapping that acts as a FIFO queue
        if queries:
            queue[root_node]["queries"] = queries
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
            # NOTE: global debug attributes will override the value specified in each node's params dictionary.
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
                    node_output, stream_id = self.graph.nodes[node_id]["component"]._dispatch_run_batch(**node_input)
                except Exception as e:
                    # The input might be a really large object with thousands of embeddings.
                    # If you really want to see it, raise the log level.
                    logger.debug("Exception while running node '%s' with input %s", node_id, node_input)
                    raise Exception(
                        f"Exception while running node '{node_id}': {e}\nEnable debug logging to see the data that was passed when the pipeline failed."
                    ) from e
                queue.pop(node_id)

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
                    for n in next_nodes:
                        if queue.get(n):  # concatenate inputs if it's a join node
                            existing_input = queue[n]
                            if "inputs" not in existing_input.keys():
                                updated_input: Dict = {"inputs": [existing_input, node_output], "params": params}
                                if queries:
                                    updated_input["queries"] = queries
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

        # increase counter of how many queries/documents have been processed by the pipeline
        if queries:
            self.run_total += len(queries)
        elif documents:
            self.run_total += len(documents)
        else:
            self.run_total += 1

        self.send_pipeline_event_if_needed()
        return node_output

    @classmethod
    def eval_beir(
        cls,
        index_pipeline: Pipeline,
        query_pipeline: Pipeline,
        index_params: dict = {},
        query_params: dict = {},
        dataset: str = "scifact",
        dataset_dir: Path = Path("."),
        num_documents: Optional[int] = None,
        top_k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        keep_index: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Runs information retrieval evaluation of a pipeline using BEIR on a specified BEIR dataset.
        See https://github.com/beir-cellar/beir for more information.

        :param index_pipeline: The indexing pipeline to use.
        :param query_pipeline: The query pipeline to evaluate.
        :param index_params: The params to use during indexing (see pipeline.run's params).
        :param query_params: The params to use during querying (see pipeline.run's params).
        :param dataset: The BEIR dataset to use.
        :param dataset_dir: The directory to store the dataset to.
        :param num_documents: Maximum number of documents to load from given dataset. If set to None (default)
                             or to a value larger than the number of documents in the dataset, the full dataset is loaded.
        :param top_k_values: The top_k values each metric will be calculated for.
        :param keep_index: Whether to keep the index after evaluation.
                           If True the index will be kept after beir evaluation. Otherwise it will be deleted immediately afterwards.
                           Defaults to False.

        Returns a tuple containing the ncdg, map, recall and precision scores.
        Each metric is represented by a dictionary containing the scores for each top_k value.
        """
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ModuleNotFoundError as e:
            raise HaystackError("beir is not installed. Please run `pip install farm-haystack[beir]`...") from e

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, dataset_dir)
        logger.info("Dataset downloaded here: %s", data_path)
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")  # or split = "train" or "dev"

        # crop dataset if `dataset_size` is provided and is valid
        if num_documents is not None and 0 < num_documents < len(corpus):
            logger.info(f"Cropping dataset from {len(corpus)} to {num_documents} documents")
            corpus = dict(itertools.islice(corpus.items(), num_documents))
            # Remove queries that don't contain the remaining documents
            corpus_ids = set(list(corpus.keys()))
            qrels_new = {}
            for query_id, document_rel_dict in qrels.items():
                document_rel_ids_intersection = list(corpus_ids & set(list(document_rel_dict.keys())))
                # If there are no remaining documents related to the query, delete the query
                if len(document_rel_ids_intersection) == 0:
                    del queries[query_id]
                # If there are remaining documents, update qrels
                else:
                    qrels_new[query_id] = {_id: qrels[query_id][_id] for _id in document_rel_ids_intersection}
            qrels = qrels_new
        elif num_documents is not None and (num_documents < 1 or num_documents > len(corpus)):
            logging.warning(
                f"'num_documents' variable should be lower than corpus length and have a positive value, but it's {num_documents}."
                " Dataset size remains unchanged."
            )

        # check index before eval
        document_store = index_pipeline.get_document_store()
        if document_store is not None:
            if document_store.get_document_count() > 0:
                raise HaystackError(f"Index '{document_store.index}' is not empty. Please provide an empty index.")

            if hasattr(document_store, "search_fields"):
                search_fields = getattr(document_store, "search_fields")
                if "name" not in search_fields:
                    logger.warning(
                        "Field 'name' is not part of your DocumentStore's search_fields. Titles won't be searchable. "
                        "Please set search_fields appropriately."
                    )

        haystack_retriever = _HaystackBeirRetrieverAdapter(
            index_pipeline=index_pipeline,
            query_pipeline=query_pipeline,
            index_params=index_params,
            query_params=query_params,
        )
        retriever = EvaluateRetrieval(haystack_retriever, k_values=top_k_values)

        # Retrieve results (format of results is identical to qrels)
        results = retriever.retrieve(corpus, queries)

        # Clean up document store
        if not keep_index and document_store is not None and document_store.index is not None:
            logger.info("Cleaning up: deleting index '%s' ...", document_store.index)
            document_store.delete_index(document_store.index)

        # Evaluate your retrieval using NDCG@k, MAP@K ...
        logger.info("Retriever evaluation for k in: %s", retriever.k_values)
        ndcg, map_, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        return ndcg, map_, recall, precision

    @classmethod
    def execute_eval_run(
        cls,
        index_pipeline: Pipeline,
        query_pipeline: Pipeline,
        evaluation_set_labels: List[MultiLabel],
        corpus_file_paths: List[str],
        experiment_name: str,
        experiment_run_name: str,
        experiment_tracking_tool: Literal["mlflow", None] = None,
        experiment_tracking_uri: Optional[str] = None,
        corpus_file_metas: Optional[List[Dict[str, Any]]] = None,
        corpus_meta: Dict[str, Any] = {},
        evaluation_set_meta: Dict[str, Any] = {},
        pipeline_meta: Dict[str, Any] = {},
        index_params: dict = {},
        query_params: dict = {},
        sas_model_name_or_path: Optional[str] = None,
        sas_batch_size: int = 32,
        sas_use_gpu: bool = True,
        use_batch_mode: bool = False,
        add_isolated_node_eval: bool = False,
        reuse_index: bool = False,
        custom_document_id_field: Optional[str] = None,
        document_scope: Literal[
            "document_id",
            "context",
            "document_id_and_context",
            "document_id_or_context",
            "answer",
            "document_id_or_answer",
        ] = "document_id_or_answer",
        answer_scope: Literal["any", "context", "document_id", "document_id_and_context"] = "any",
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
    ) -> EvaluationResult:
        """
        Starts an experiment run that first indexes the specified files (forming a corpus) using the index pipeline
        and subsequently evaluates the query pipeline on the provided labels (forming an evaluation set) using pipeline.eval().
        Parameters and results (metrics and predictions) of the run are tracked by an experiment tracking tool for further analysis.
        You can specify the experiment tracking tool by setting the params `experiment_tracking_tool` and `experiment_tracking_uri`
        or by passing a (custom) tracking head to Tracker.set_tracking_head().
        Note, that `experiment_tracking_tool` only supports `mlflow` currently.

        For easier comparison you can pass additional metadata regarding corpus (corpus_meta), evaluation set (evaluation_set_meta) and pipelines (pipeline_meta).
        E.g. you can give them names or ids to identify them across experiment runs.

        This method executes an experiment run. Each experiment run is part of at least one experiment.
        An experiment typically consists of multiple runs to be compared (e.g. using different retrievers in query pipeline).
        Experiment tracking tools usually share the same concepts of experiments and provide additional functionality to easily compare runs across experiments.

        E.g. you can call execute_eval_run() multiple times with different retrievers in your query pipeline and compare the runs in mlflow:

        ```python
        for retriever_type, query_pipeline in zip(["sparse", "dpr", "embedding"], [sparse_pipe, dpr_pipe, embedding_pipe]):
            eval_result = Pipeline.execute_eval_run(
                index_pipeline=index_pipeline,
                query_pipeline=query_pipeline,
                evaluation_set_labels=labels,
                corpus_file_paths=file_paths,
                corpus_file_metas=file_metas,
                experiment_tracking_tool="mlflow",
                experiment_tracking_uri="http://localhost:5000",
                experiment_name="my-retriever-experiment",
                experiment_run_name=f"run_{retriever_type}",
                pipeline_meta={"name": f"my-pipeline-{retriever_type}"},
                evaluation_set_meta={"name": "my-evalset"},
                corpus_meta={"name": "my-corpus"}.
                reuse_index=False
            )
        ```

        :param index_pipeline: The indexing pipeline to use.
        :param query_pipeline: The query pipeline to evaluate.
        :param evaluation_set_labels: The labels to evaluate on forming an evaluation set.
        :param corpus_file_paths: The files to be indexed and searched during evaluation forming a corpus.
        :param experiment_name: The name of the experiment
        :param experiment_run_name: The name of the experiment run
        :param experiment_tracking_tool: The experiment tracking tool to be used. Currently we only support "mlflow".
                                         If left unset the current TrackingHead specified by Tracker.set_tracking_head() will be used.
        :param experiment_tracking_uri: The uri of the experiment tracking server to be used. Must be specified if experiment_tracking_tool is set.
                                        You can use deepset's public mlflow server via https://public-mlflow.deepset.ai/.
                                        Note, that artifact logging (e.g. Pipeline YAML or evaluation result CSVs) are currently not allowed on deepset's public mlflow server as this might expose sensitive data.
        :param corpus_file_metas: The optional metadata to be stored for each corpus file (e.g. title).
        :param corpus_meta: Metadata about the corpus to track (e.g. name, date, author, version).
        :param evaluation_set_meta: Metadata about the evalset to track (e.g. name, date, author, version).
        :param pipeline_meta: Metadata about the pipelines to track (e.g. name, author, version).
        :param index_params: The params to use during indexing (see pipeline.run's params).
        :param query_params: The params to use during querying (see pipeline.run's params).
        :param sas_model_name_or_path: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
                    The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
                    Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture with being close to 1.
                    More info in the paper: https://arxiv.org/abs/2108.06130
                    Here are some guidelines regarding the models that you can use:
                    - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                    The return type of the encoder needs to be a single prediction score (as opposed to multiple scores).
                    If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
                    - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
                    - Large model for German only: "deepset/gbert-large-sts"
        :param sas_batch_size: Number of prediction label pairs to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.
        :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                            Falls back to CPU if no GPU is available.
        :param use_batch_mode: Whether to use batches for pipeline executions or single queries (default).
        :param add_isolated_node_eval: If set to True, in addition to the integrated evaluation of the pipeline, each node is evaluated in isolated evaluation mode.
                    The isolated mode shows you how each node is performing on its own and helps to understand the bottlenecks of a pipeline in terms of output quality of each individual node.
                    If a node performs much better in the isolated evaluation than in the integrated evaluation, the previous node needs to be optimized to improve the pipeline's performance.
                    If a node's performance is similar in both modes, this node itself needs to be optimized to improve the pipeline's performance.
                    The isolated evaluation calculates the upper bound of each node's evaluation metrics under the assumption that it received perfect inputs from the previous node.
                    To this end, labels are used as input to the node instead of the output of the previous node in the pipeline.
                    The generated dataframes in the EvaluationResult then contain additional rows, which can be distinguished from the integrated evaluation results based on the
                    values "integrated" or "isolated" in the column "eval_mode" and the evaluation report then additionally lists the upper bound of each node's evaluation metrics.
        :param reuse_index: Whether to reuse existing non-empty index and to keep the index after evaluation.
                           If True the index will be kept after evaluation and no indexing will take place if index has already documents. Otherwise it will be deleted immediately afterwards.
                           Defaults to False.
        :param custom_document_id_field: Custom field name within `Document`'s `meta` which identifies the document and is being used as criterion for matching documents to labels during evaluation.
                                         This is especially useful if you want to match documents on other criteria (e.g. file names) than the default document ids as these could be heavily influenced by preprocessing.
                                         If not set (default) the `Document`'s `id` is being used as criterion for matching documents to labels.
        :param document_scope: A criterion for deciding whether documents are relevant or not.
            You can select between:
            - 'document_id': Specifies that the document ID must match. You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
                    A typical use case is Document Retrieval.
            - 'context': Specifies that the content of the document must match. Uses fuzzy matching (see `context_matching_...` params).
                    A typical use case is Document-Independent Passage Retrieval.
            - 'document_id_and_context': A Boolean operation specifying that both `'document_id' AND 'context'` must match.
                    A typical use case is Document-Specific Passage Retrieval.
            - 'document_id_or_context': A Boolean operation specifying that either `'document_id' OR 'context'` must match.
                    A typical use case is Document Retrieval having sparse context labels.
            - 'answer': Specifies that the document contents must include the answer. The selected `answer_scope` is enforced automatically.
                    A typical use case is Question Answering.
            - 'document_id_or_answer' (default): A Boolean operation specifying that either `'document_id' OR 'answer'` must match.
                    This is intended to be a proper default value in order to support both main use cases:
                    - Document Retrieval
                    - Question Answering
            The default value is 'document_id_or_answer'.
        :param answer_scope: Specifies the scope in which a matching answer is considered correct.
            You can select between:
            - 'any' (default): Any matching answer is considered correct.
            - 'context': The answer is only considered correct if its context matches as well.
                    Uses fuzzy matching (see `context_matching_...` params).
            - 'document_id': The answer is only considered correct if its document ID matches as well.
                    You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
            - 'document_id_and_context': The answer is only considered correct if its document ID and its context match as well.
            The default value is 'any'.
            In Question Answering, to enforce that the retrieved document is considered correct whenever the answer is correct, set `document_scope` to 'answer' or 'document_id_or_answer'.
        :param context_matching_min_length: The minimum string length context and candidate need to have in order to be scored.
                           Returns 0.0 otherwise.
        :param context_matching_boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
        :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]
        """
        if experiment_tracking_tool is not None:
            tracking_head_cls = TRACKING_TOOL_TO_HEAD.get(experiment_tracking_tool, None)
            if tracking_head_cls is None:
                raise HaystackError(
                    f"Please specify a valid experiment_tracking_tool. Possible values are: {TRACKING_TOOL_TO_HEAD.keys()}"
                )
            if experiment_tracking_uri is None:
                raise HaystackError(f"experiment_tracking_uri must be specified if experiment_tracking_tool is set.")
            tracking_head = tracking_head_cls(tracking_uri=experiment_tracking_uri)
            tracker.set_tracking_head(tracking_head)

        try:
            tracker.init_experiment(
                experiment_name=experiment_name, run_name=experiment_run_name, tags={experiment_name: "True"}
            )
            tracker.track_params(
                {
                    "evaluation_set_label_count": len(evaluation_set_labels),
                    "evaluation_set": evaluation_set_meta,
                    "sas_model_name_or_path": sas_model_name_or_path,
                    "sas_batch_size": sas_batch_size,
                    "sas_use_gpu": sas_use_gpu,
                    "pipeline_index_params": index_params,
                    "pipeline_query_params": query_params,
                    "pipeline": pipeline_meta,
                    "corpus_file_count": len(corpus_file_paths),
                    "corpus": corpus_meta,
                    "type": "offline/evaluation",
                    "document_scope": document_scope,
                    "answer_scope": answer_scope,
                    "custom_document_id_field": custom_document_id_field,
                    "context_matching_min_length": context_matching_min_length,
                    "context_matching_boost_split_overlaps": context_matching_boost_split_overlaps,
                    "context_matching_threshold": context_matching_threshold,
                }
            )

            # check index before eval
            document_store = index_pipeline.get_document_store()
            if document_store is None:
                raise HaystackError(f"Document store not found. Please provide pipelines with proper document store.")
            document_count = document_store.get_document_count()

            if document_count > 0:
                if not reuse_index:
                    raise HaystackError(f"Index '{document_store.index}' is not empty. Please provide an empty index.")
            else:
                logger.info("indexing %s documents...", len(corpus_file_paths))
                index_pipeline.run(file_paths=corpus_file_paths, meta=corpus_file_metas, params=index_params)
                document_count = document_store.get_document_count()
                logger.info("indexing %s files to %s documents finished.", len(evaluation_set_labels), document_count)

            tracker.track_params({"pipeline_index_document_count": document_count})

            if use_batch_mode:
                eval_result = query_pipeline.eval_batch(
                    labels=evaluation_set_labels,
                    params=query_params,
                    sas_model_name_or_path=sas_model_name_or_path,
                    sas_batch_size=sas_batch_size,
                    sas_use_gpu=sas_use_gpu,
                    add_isolated_node_eval=add_isolated_node_eval,
                    custom_document_id_field=custom_document_id_field,
                    context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
                    context_matching_min_length=context_matching_min_length,
                    context_matching_threshold=context_matching_threshold,
                )
            else:
                eval_result = query_pipeline.eval(
                    labels=evaluation_set_labels,
                    params=query_params,
                    sas_model_name_or_path=sas_model_name_or_path,
                    sas_batch_size=sas_batch_size,
                    sas_use_gpu=sas_use_gpu,
                    add_isolated_node_eval=add_isolated_node_eval,
                    custom_document_id_field=custom_document_id_field,
                    context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
                    context_matching_min_length=context_matching_min_length,
                    context_matching_threshold=context_matching_threshold,
                )

            integrated_metrics = eval_result.calculate_metrics(document_scope=document_scope, answer_scope=answer_scope)
            integrated_top_1_metrics = eval_result.calculate_metrics(
                simulated_top_k_reader=1, document_scope=document_scope, answer_scope=answer_scope
            )
            metrics = {"integrated": integrated_metrics, "integrated_top_1": integrated_top_1_metrics}
            if add_isolated_node_eval:
                isolated_metrics = eval_result.calculate_metrics(
                    eval_mode="isolated", document_scope=document_scope, answer_scope=answer_scope
                )
                isolated_top_1_metrics = eval_result.calculate_metrics(
                    eval_mode="isolated",
                    simulated_top_k_reader=1,
                    document_scope=document_scope,
                    answer_scope=answer_scope,
                )
                metrics["isolated"] = isolated_metrics
                metrics["isolated_top_1"] = isolated_top_1_metrics
            tracker.track_metrics(metrics, step=0)

            with tempfile.TemporaryDirectory() as temp_dir:
                eval_result_dir = Path(temp_dir) / "eval_result"
                eval_result_dir.mkdir(exist_ok=True)
                eval_result.save(out_dir=eval_result_dir)
                tracker.track_artifacts(eval_result_dir, artifact_path="eval_result")
                with open(Path(temp_dir) / "pipelines.yaml", "w") as outfile:
                    index_config = index_pipeline.get_config()
                    query_config = query_pipeline.get_config()
                    components = list(
                        {c["name"]: c for c in (index_config["components"] + query_config["components"])}.values()
                    )
                    pipelines = index_config["pipelines"] + query_config["pipelines"]
                    config = {"version": index_config["version"], "components": components, "pipelines": pipelines}
                    yaml.dump(config, outfile, default_flow_style=False)
                tracker.track_artifacts(temp_dir)

            # Clean up document store
            if not reuse_index and document_store.index is not None:
                logger.info("Cleaning up: deleting index '%s'...", document_store.index)
                document_store.delete_index(document_store.index)

        finally:
            tracker.end_run()

        return eval_result

    @send_event
    def eval(
        self,
        labels: List[MultiLabel],
        documents: Optional[List[List[Document]]] = None,
        params: Optional[dict] = None,
        sas_model_name_or_path: Optional[str] = None,
        sas_batch_size: int = 32,
        sas_use_gpu: bool = True,
        add_isolated_node_eval: bool = False,
        custom_document_id_field: Optional[str] = None,
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
        use_auth_token: Optional[Union[str, bool]] = None,
    ) -> EvaluationResult:
        """
        Evaluates the pipeline by running the pipeline once per query in debug mode
        and putting together all data that is needed for evaluation, e.g. calculating metrics.

        If you want to calculate SAS (Semantic Answer Similarity) metrics, you have to specify `sas_model_name_or_path`.

        You will be able to control the scope within which an answer or a document is considered correct afterwards (See `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).
        Some of these scopes require additional information that already needs to be specified during `eval()`:
        - `custom_document_id_field` param to select a custom document ID from document's meta data for ID matching (only affects 'document_id' scopes)
        - `context_matching_...` param to fine-tune the fuzzy matching mechanism that determines whether some text contexts match each other (only affects 'context' scopes, default values should work most of the time)

        :param labels: The labels to evaluate on
        :param documents: List of List of Document that the first node in the pipeline should get as input per multilabel. Can be used to evaluate a pipeline that consists of a reader without a retriever.
        :param params: A dictionary of parameters that you want to pass to the nodes.
                    If you want to pass a param to all nodes, you can just use: {"top_k":10}
                    If you want to pass it to targeted nodes, you can do:
                    {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
        :param sas_model_name_or_path: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
                    The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
                    Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture with being close to 1.
                    More info in the paper: https://arxiv.org/abs/2108.06130
                    Here are some guidelines regarding the models that you can use:
                    - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                    The return type of the encoder needs to be a single prediction score (as opposed to multiple scores).
                    If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
                    - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
                    - Large model for German only: "deepset/gbert-large-sts"
        :param sas_batch_size: Number of prediction label pairs to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.
        :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                            Falls back to CPU if no GPU is available.
        :param add_isolated_node_eval: If set to True, in addition to the integrated evaluation of the pipeline, each node is evaluated in isolated evaluation mode.
                    The isolated mode shows you how each node is performing on its own and helps to understand the bottlenecks of a pipeline in terms of output quality of each individual node.
                    If a node performs much better in the isolated evaluation than in the integrated evaluation, the previous node needs to be optimized to improve the pipeline's performance.
                    If a node's performance is similar in both modes, this node itself needs to be optimized to improve the pipeline's performance.
                    The isolated evaluation calculates the upper bound of each node's evaluation metrics under the assumption that it received perfect inputs from the previous node.
                    To this end, labels are used as input to the node instead of the output of the previous node in the pipeline.
                    The generated dataframes in the EvaluationResult then contain additional rows, which can be distinguished from the integrated evaluation results based on the
                    values "integrated" or "isolated" in the column "eval_mode" and the evaluation report then additionally lists the upper bound of each node's evaluation metrics.
        :param custom_document_id_field: Custom field name within `Document`'s `meta` which identifies the document and is being used as criterion for matching documents to labels during evaluation.
                                         This is especially useful if you want to match documents on other criteria (e.g. file names) than the default document ids as these could be heavily influenced by preprocessing.
                                         If not set (default) the `Document`'s `id` is being used as criterion for matching documents to labels.
        :param context_matching_min_length: The minimum string length context and candidate need to have in order to be scored.
                           Returns 0.0 otherwise.
        :param context_matching_boost_split_overlaps: Whether to boost split overlaps (e.g. [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries
                                 we cut the context on the same side, recalculate the score and take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
        :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100]
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        eval_result = EvaluationResult()
        if add_isolated_node_eval:
            params = {} if params is None else params.copy()
            params["add_isolated_node_eval"] = True

        # if documents is None, set docs_per_label to None for each label
        for docs_per_label, label in zip(documents or [None] * len(labels), labels):  # type: ignore
            params_per_label = copy.deepcopy(params)
            # If the label contains a filter, the filter is applied unless documents are already given
            if label.filters is not None and documents is None:
                if params_per_label is None:
                    params_per_label = {"filters": label.filters}
                else:
                    # join both filters and overwrite filters in params with filters in labels
                    params_per_label["filters"] = {**params_per_label.get("filters", {}), **label.filters}
            predictions = self.run(
                query=label.query, labels=label, documents=docs_per_label, params=params_per_label, debug=True
            )

            for node_name in predictions["_debug"].keys():
                node_output = predictions["_debug"][node_name]["output"]
                df = self._build_eval_dataframe(
                    queries=[label.query],
                    query_labels_per_query=[label],
                    node_name=node_name,
                    node_output=node_output,
                    custom_document_id_field=custom_document_id_field,
                    context_matching_threshold=context_matching_threshold,
                    context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
                    context_matching_min_length=context_matching_min_length,
                )
                eval_result.append(node_name, df)

        eval_result = self._add_sas_to_eval_result(
            sas_model_name_or_path=sas_model_name_or_path,
            sas_batch_size=sas_batch_size,
            sas_use_gpu=sas_use_gpu,
            context_matching_threshold=context_matching_threshold,
            eval_result=eval_result,
            use_auth_token=use_auth_token,
        )
        # reorder columns for better qualitative evaluation
        eval_result = self._reorder_columns_in_eval_result(eval_result=eval_result)

        return eval_result

    @send_event
    def eval_batch(
        self,
        labels: List[MultiLabel],
        documents: Optional[List[List[Document]]] = None,
        params: Optional[dict] = None,
        sas_model_name_or_path: Optional[str] = None,
        sas_batch_size: int = 32,
        sas_use_gpu: bool = True,
        add_isolated_node_eval: bool = False,
        custom_document_id_field: Optional[str] = None,
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
        use_auth_token: Optional[Union[str, bool]] = None,
    ) -> EvaluationResult:
        """
        Evaluates the pipeline by running it in batches in the debug mode
        and putting together all data that are needed for evaluation, for example, calculating metrics.

        To calculate SAS (Semantic Answer Similarity) metrics, specify `sas_model_name_or_path`.

        You can control the scope within which an answer or a document is considered correct afterwards (see `document_scope` and `answer_scope` params in `EvaluationResult.calculate_metrics()`).
        For some of these scopes, you need to add the following information during `eval()`:
        - `custom_document_id_field` parameter to select a custom document ID from document's metadata for ID matching (only affects 'document_id' scopes).
        - `context_matching_...` parameter to fine-tune the fuzzy matching mechanism that determines whether text contexts match each other (only affects 'context' scopes, default values should work most of the time).

        :param labels: The labels to evaluate on.
        :param documents: List of List of Document that the first node in the pipeline gets as input per multilabel. You can use it to evaluate a pipeline that consists of a reader without a retriever.
        :param params: Dictionary of parameters to be dispatched to the nodes.
                    To pass a parameter to all nodes, just use: {"top_k":10}.
                    To pass a parametrer to targeted nodes, you can type:
                    {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
        :param sas_model_name_or_path: Name or path of the SAS model. If you specify the path, the model is used to calculate the similarity between predictions and labels and to generate the SAS metric.
                    The SAS metric correlates better with the human judgment of correct answers as it does not rely on string overlaps.
                    Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture.
                    If you want to learn more, have a look at the [Semantic Answer Similarity for Evaluating Question Answering Models](https://arxiv.org/abs/2108.06130) paper.
                    Models:
                    - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                    The return type of the encoder needs to be a single prediction score (as opposed to multiple scores).
                    When using custom cross encoders, ensure they work with the `sentence_transformers.CrossEncoder` class.
                    - This is a good default model for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2."
                    - A large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large".
                    - A large model for German only: "deepset/gbert-large-sts".
        :param sas_batch_size: The number of prediction label pairs you want to encode at once by CrossEncoder or SentenceTransformer while calculating SAS.
        :param sas_use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                            It uses CPU if no GPU is available.
        :param add_isolated_node_eval: If set to True, in addition to the integrated evaluation of the pipeline, each node is evaluated in isolated evaluation mode.
                    The isolated mode shows you how each node is performing on its own and helps to understand the bottlenecks of a pipeline in terms of output quality of each individual node.
                    If a node performs much better in the isolated evaluation than in the integrated evaluation, it means you should optimize the preceding node to improve the pipeline's performance.
                    If a node's performance is similar in both modes, it means you should optimize this node itself to improve the pipeline's performance.
                    The isolated evaluation calculates the upper bound of each node's evaluation metrics, assuming it received perfect inputs from the previous node.
                    To achieve this, the isolated evaluation uses labels as input to the node instead of the output of the previous node in the pipeline.
                    The generated dataframes in the EvaluationResult then contain additional rows, which you can tell apart from the integrated evaluation results based on the
                    values "integrated" or "isolated" in the column "eval_mode". The evaluation report then additionally lists the upper bound of each node's evaluation metrics.
        :param custom_document_id_field: Custom field name within `Document`'s `meta` which identifies the document. This field is used as a criterion for matching documents to labels during evaluation.
                                         This is especially useful if you want to match documents on other criteria (for example, file names) than the default document IDs as these could be heavily influenced by preprocessing.
                                         If you don't set any value, the default `Document`'s `id` is used as a criterion for matching documents to labels.
        :param context_matching_min_length: The minimum string length context and candidate need to have in order to be scored.
                           Returns 0.0 otherwise.
        :param context_matching_boost_split_overlaps: Whether to boost split overlaps (for example, [AB] <-> [BC]) that result from different preprocessing params.
                                 If we detect that the score is near a half match and the matching part of the candidate is at its boundaries,
                                 we cut the context on the same side, recalculate the score and, take the mean of both.
                                 Thus [AB] <-> [BC] (score ~50) gets recalculated with B <-> B (score ~100) scoring ~75 in total.
        :param context_matching_threshold: Score threshold that candidates must surpass to be included into the result list. Range: [0,100].
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        eval_result = EvaluationResult()
        if add_isolated_node_eval:
            params = {} if params is None else params.copy()
            params["add_isolated_node_eval"] = True

        predictions_batches = self.run_batch(
            queries=[label.query for label in labels], labels=labels, documents=documents, params=params, debug=True
        )

        for node_name in predictions_batches["_debug"].keys():
            node_output = predictions_batches["_debug"][node_name]["output"]
            df = self._build_eval_dataframe(
                queries=predictions_batches["queries"],
                query_labels_per_query=predictions_batches["labels"],
                node_name=node_name,
                node_output=node_output,
                custom_document_id_field=custom_document_id_field,
                context_matching_threshold=context_matching_threshold,
                context_matching_boost_split_overlaps=context_matching_boost_split_overlaps,
                context_matching_min_length=context_matching_min_length,
            )
            eval_result.append(node_name, df)

        eval_result = self._add_sas_to_eval_result(
            sas_model_name_or_path=sas_model_name_or_path,
            sas_batch_size=sas_batch_size,
            sas_use_gpu=sas_use_gpu,
            context_matching_threshold=context_matching_threshold,
            eval_result=eval_result,
            use_auth_token=use_auth_token,
        )
        # reorder columns for better qualitative evaluation
        eval_result = self._reorder_columns_in_eval_result(eval_result=eval_result)

        return eval_result

    def _add_sas_to_eval_result(
        self,
        sas_model_name_or_path: Optional[str],
        sas_batch_size: int,
        sas_use_gpu: bool,
        context_matching_threshold: float,
        eval_result: EvaluationResult,
        use_auth_token: Optional[Union[str, bool]] = None,
    ) -> EvaluationResult:
        # add sas values in batch mode for whole Dataframe
        # this is way faster than if we calculate it for each query separately
        if sas_model_name_or_path is not None:
            for df in eval_result.node_results.values():
                if len(df[df["type"] == "answer"]) > 0:
                    gold_labels = df["gold_answers"].values
                    predictions = [[a] for a in df["answer"].values]
                    sas, _, pred_label_sas_grid = semantic_answer_similarity(
                        predictions=predictions,
                        gold_labels=gold_labels,
                        sas_model_name_or_path=sas_model_name_or_path,
                        batch_size=sas_batch_size,
                        use_gpu=sas_use_gpu,
                        use_auth_token=use_auth_token,
                    )
                    df["sas"] = sas
                    df["gold_answers_sas"] = [
                        gold_answers_sas_per_pred[0] for gold_answers_sas_per_pred in pred_label_sas_grid
                    ]
                    df.map_rows = partial(df.apply, axis=1)
                    df["sas_context_scope"] = df.map_rows(
                        lambda row: max(
                            sas
                            for sas, sim in zip(
                                row["gold_answers_sas"] + [0.0], row["gold_contexts_similarity"] + [100]
                            )
                            if sim > context_matching_threshold
                        )
                    )
                    df["sas_document_id_scope"] = df.map_rows(
                        lambda row: max(
                            sas
                            for sas, doc_match in zip(
                                row["gold_answers_sas"] + [0.0], row["gold_documents_id_match"] + [1.0]
                            )
                            if doc_match == 1.0
                        )
                    )
                    df["sas_document_id_and_context_scope"] = df.map_rows(
                        lambda row: max(
                            sas
                            for sas, sim, doc_match in zip(
                                row["gold_answers_sas"] + [0.0],
                                row["gold_contexts_similarity"] + [100],
                                row["gold_documents_id_match"] + [1.0],
                            )
                            if sim > context_matching_threshold and doc_match == 1.0
                        )
                    )

        return eval_result

    def _reorder_columns_in_eval_result(self, eval_result: EvaluationResult) -> EvaluationResult:
        desired_col_order = [
            "multilabel_id",  # generic
            "query",  # generic
            "filters",  # generic
            "gold_answers",  # answer-specific
            "answer",  # answer-specific
            "context",  # generic
            "exact_match",  # answer-specific
            "f1",  # answer-specific
            "sas",  # answer-specific
            "exact_match_context_scope",  # answer-specific
            "f1_context_scope",  # answer-specific
            "sas_context_scope",  # answer-specific
            "exact_match_document_id_scope",  # answer-specific
            "f1_document_id_scope",  # answer-specific
            "sas_document_id_scope",  # answer-specific
            "exact_match_document_id_and_context_scope",  # answer-specific
            "f1_document_id_and_context_scope",  # answer-specific
            "sas_document_id_and_context_scope",  # answer-specific
            "gold_contexts",  # generic
            "gold_id_match",  # doc-specific
            "context_match",  # doc-specific
            "answer_match",  # doc-specific
            "gold_id_or_answer_match",  # doc-specific
            "gold_id_and_answer_match",  # doc-specific
            "gold_id_or_context_match",  # doc-specific
            "gold_id_and_context_match",  # doc-specific
            "gold_id_and_context_and_answer_match",  # doc-specific
            "context_and_answer_match",  # doc-specific
            "rank",  # generic
            "document_id",  # generic
            "gold_document_ids",  # generic
            "custom_document_id",  # generic
            "gold_custom_document_ids",  # generic
            "offsets_in_document",  # answer-specific
            "gold_offsets_in_documents",  # answer-specific
            "offsets_in_context",  # answer-specific
            "gold_offsets_in_contexts",  # answer-specific
            "gold_answers_exact_match",  # answer-specific
            "gold_answers_f1",  # answer-specific
            "gold_answers_sas",  # answer-specific
            "gold_documents_id_match",  # generic
            "gold_contexts_similarity",  # generic
            "gold_answers_match",  # doc-specific
            "type",  # generic
            "node",  # generic
            "eval_mode",  # generic
        ]
        for key, df in eval_result.node_results.items():
            eval_result.node_results[key] = self._reorder_columns(df, desired_col_order)

        return eval_result

    def _reorder_columns(self, df: DataFrame, desired_order: List[str]) -> DataFrame:
        filtered_order = [col for col in desired_order if col in df.columns]
        missing_columns = [col for col in df.columns if col not in desired_order]
        reordered_columns = filtered_order + missing_columns
        assert len(reordered_columns) == len(df.columns)
        return df.reindex(columns=reordered_columns)

    def _build_eval_dataframe(
        self,
        queries: List[str],
        query_labels_per_query: List[MultiLabel],
        node_name: str,
        node_output: dict,
        custom_document_id_field: Optional[str] = None,
        context_matching_min_length: int = 100,
        context_matching_boost_split_overlaps: bool = True,
        context_matching_threshold: float = 65.0,
    ) -> DataFrame:
        """
        Builds a Dataframe for each query from which evaluation metrics can be calculated.
        Currently only answer or document returning nodes are supported, returns None otherwise.

        Each row contains either an answer or a document that has been retrieved during evaluation.
        Rows are being enriched with basic infos like rank, query, type or node.
        Additional answer or document specific evaluation infos like gold labels
        and metrics depicting whether the row matches the gold labels are included, too.
        """
        # Disable all the cell-var-from-loop violations in this function
        # pylint: disable=cell-var-from-loop

        partial_dfs = []
        for i, (query, query_labels) in enumerate(zip(queries, query_labels_per_query)):

            if query_labels is None or query_labels.labels is None:
                logger.warning("There is no label for query '%s'. Query will be omitted.", query)
                continue

            # remarks for no_answers:
            # Single 'no_answer'-labels are not contained in MultiLabel aggregates.
            # If all labels are no_answers, MultiLabel.answers will be [""] and the other aggregates []
            gold_answers = query_labels.answers
            gold_offsets_in_documents = query_labels.offsets_in_documents
            gold_offsets_in_contexts = query_labels.offsets_in_contexts
            gold_document_ids = query_labels.document_ids
            gold_custom_document_ids = (
                [l.document.meta[custom_document_id_field] for l in query_labels.labels if not l.no_answer]
                if custom_document_id_field is not None
                else []
            )
            gold_contexts = query_labels.contexts

            # if node returned answers, include answer specific info:
            # - the answer returned itself
            # - the document_id the answer was found in
            # - the position or offsets within the document the answer was found
            # - the surrounding context of the answer within the document
            # - the gold answers
            # - the position or offsets of the gold answer within the document
            # - the gold document ids containing the answer
            # - the exact_match metric depicting if the answer exactly matches the gold label
            # - the f1 metric depicting how well the answer overlaps with the gold label on token basis
            # - the sas metric depicting how well the answer matches the gold label on a semantic basis.
            #   this will be calculated on all queries in eval() for performance reasons if a sas model has been provided

            for field_name in ["answers", "answers_isolated"]:
                df_answers = pd.DataFrame()
                answers = node_output.get(field_name, None)
                if answers is not None:
                    if i < len(answers) and isinstance(answers[i], list):
                        # answers_isolated refers to only one relevant document and thus only a list of answers
                        # answers refers to multiple relevant documents and thus multiple lists of lists of answers
                        answers = answers[i]
                    if len(answers) == 0:
                        # add no_answer if there was no answer retrieved, so query does not get lost in dataframe
                        answers = [
                            Answer(
                                answer="",
                                offsets_in_document=[Span(start=0, end=0)],
                                offsets_in_context=[Span(start=0, end=0)],
                            )
                        ]
                    answer_cols_to_keep = [
                        "answer",
                        "document_id",
                        "offsets_in_document",
                        "offsets_in_context",
                        "context",
                    ]
                    df_answers = pd.DataFrame(answers, columns=answer_cols_to_keep)
                    df_answers.map_rows = partial(df_answers.apply, axis=1)
                    df_answers["rank"] = np.arange(1, len(df_answers) + 1)
                    df_answers["gold_answers"] = [gold_answers] * len(df_answers)
                    df_answers["gold_offsets_in_documents"] = [gold_offsets_in_documents] * len(df_answers)
                    df_answers["gold_offsets_in_contexts"] = [gold_offsets_in_contexts] * len(df_answers)
                    df_answers["gold_document_ids"] = [gold_document_ids] * len(df_answers)
                    df_answers["gold_contexts"] = [gold_contexts] * len(df_answers)
                    df_answers["gold_answers_exact_match"] = df_answers.map_rows(
                        lambda row: [calculate_em_str(gold_answer, row["answer"]) for gold_answer in gold_answers]
                    )
                    df_answers["gold_answers_f1"] = df_answers.map_rows(
                        lambda row: [calculate_f1_str(gold_answer, row["answer"]) for gold_answer in gold_answers]
                    )
                    df_answers["gold_contexts_similarity"] = df_answers.map_rows(
                        lambda row: [
                            calculate_context_similarity(
                                str(gold_context),  # could be dataframe
                                str(row["context"]) if row["context"] is not None else "",  # could be dataframe
                                min_length=context_matching_min_length,
                                boost_split_overlaps=context_matching_boost_split_overlaps,
                            )
                            for gold_context in gold_contexts
                        ]
                    )
                    df_answers["gold_documents_id_match"] = df_answers.map_rows(
                        lambda row: [1.0 if row["document_id"] == gold_id else 0.0 for gold_id in gold_document_ids]
                    )

                    if custom_document_id_field is not None:
                        df_answers["gold_custom_document_ids"] = [gold_custom_document_ids] * len(df_answers)
                        df_answers["custom_document_id"] = [
                            answer.meta.get(custom_document_id_field, "") for answer in answers
                        ]
                        df_answers["gold_documents_id_match"] = df_answers.map_rows(
                            lambda row: [
                                1.0 if row["custom_document_id"] == gold_custom_id else 0.0
                                for gold_custom_id in gold_custom_document_ids
                            ]
                        )

                    # answer_scope: any
                    df_answers["exact_match"] = df_answers.map_rows(
                        lambda row: max(row["gold_answers_exact_match"] + [0.0])
                    )
                    df_answers["f1"] = df_answers.map_rows(lambda row: max(row["gold_answers_f1"] + [0.0]))

                    # answer_scope: context
                    df_answers["exact_match_context_scope"] = df_answers.map_rows(
                        lambda row: max(
                            em
                            for em, sim in zip(
                                row["gold_answers_exact_match"] + [0.0], row["gold_contexts_similarity"] + [100]
                            )
                            if sim > context_matching_threshold
                        )
                    )
                    df_answers["f1_context_scope"] = df_answers.map_rows(
                        lambda row: max(
                            f1
                            for f1, sim in zip(row["gold_answers_f1"] + [0.0], row["gold_contexts_similarity"] + [100])
                            if sim > context_matching_threshold
                        )
                    )

                    # answer_scope: document_id
                    df_answers["exact_match_document_id_scope"] = df_answers.map_rows(
                        lambda row: max(
                            em
                            for em, doc_match in zip(
                                row["gold_answers_exact_match"] + [0.0], row["gold_documents_id_match"] + [1.0]
                            )
                            if doc_match == 1.0
                        )
                    )
                    df_answers["f1_document_id_scope"] = df_answers.map_rows(
                        lambda row: max(
                            f1
                            for f1, doc_match in zip(
                                row["gold_answers_f1"] + [0.0], row["gold_documents_id_match"] + [1.0]
                            )
                            if doc_match == 1.0
                        )
                    )

                    # answer_scope: document_id_and_context
                    df_answers["exact_match_document_id_and_context_scope"] = df_answers.map_rows(
                        lambda row: max(
                            f1
                            for f1, sim, doc_match in zip(
                                row["gold_answers_exact_match"] + [0.0],
                                row["gold_contexts_similarity"] + [100],
                                row["gold_documents_id_match"] + [1.0],
                            )
                            if sim > context_matching_threshold and doc_match == 1.0
                        )
                    )
                    df_answers["f1_document_id_and_context_scope"] = df_answers.map_rows(
                        lambda row: max(
                            f1
                            for f1, sim, doc_match in zip(
                                row["gold_answers_f1"] + [0.0],
                                row["gold_contexts_similarity"] + [100],
                                row["gold_documents_id_match"] + [1.0],
                            )
                            if sim > context_matching_threshold and doc_match == 1.0
                        )
                    )

                # add general info
                df_answers["type"] = "answer"
                df_answers["node"] = node_name
                df_answers["multilabel_id"] = query_labels.id
                df_answers["query"] = query
                df_answers["filters"] = json.dumps(query_labels.filters, sort_keys=True).encode()
                df_answers["eval_mode"] = "isolated" if "isolated" in field_name else "integrated"
                partial_dfs.append(df_answers)

            # if node returned documents, include document specific info:
            # - the document_id
            # - the content of the document
            # - the gold document ids
            # - the gold document contents
            # - the gold_id_match metric depicting whether one of the gold document ids matches the document
            # - the answer_match metric depicting whether the document contains the answer
            # - the gold_id_or_answer_match metric depicting whether one of the former two conditions are met
            for field_name in ["documents", "documents_isolated"]:
                df_docs = pd.DataFrame()
                documents = node_output.get(field_name, None)
                if documents is not None:
                    if i < len(documents) and isinstance(documents[i], list):
                        documents = documents[i]
                    if len(documents) == 0:
                        # add dummy document if there was no document retrieved, so query does not get lost in dataframe
                        documents = [Document(content="", id="")]
                    document_cols_to_keep = ["content", "id"]
                    df_docs = pd.DataFrame(documents, columns=document_cols_to_keep)
                    df_docs.map_rows = partial(df_docs.apply, axis=1)
                    df_docs.rename(columns={"id": "document_id", "content": "context"}, inplace=True)
                    df_docs["gold_document_ids"] = [gold_document_ids] * len(df_docs)
                    df_docs["gold_contexts"] = [gold_contexts] * len(df_docs)
                    df_docs["gold_contexts_similarity"] = df_docs.map_rows(
                        lambda row: [
                            calculate_context_similarity(
                                str(gold_context) if isinstance(gold_context, pd.DataFrame) else gold_context,
                                str(row["context"])
                                if isinstance(row["context"], pd.DataFrame)
                                else row["context"] or "",
                                min_length=context_matching_min_length,
                                boost_split_overlaps=context_matching_boost_split_overlaps,
                            )
                            for gold_context in gold_contexts
                        ]
                    )
                    df_docs["gold_documents_id_match"] = df_docs.map_rows(
                        lambda row: [1.0 if row["document_id"] == gold_id else 0.0 for gold_id in gold_document_ids]
                    )

                    if custom_document_id_field is not None:
                        df_docs["gold_custom_document_ids"] = [gold_custom_document_ids] * len(df_docs)
                        df_docs["custom_document_id"] = [
                            document.meta.get(custom_document_id_field, "") for document in documents
                        ]
                        df_docs["gold_documents_id_match"] = df_docs.map_rows(
                            lambda row: [
                                1.0 if row["custom_document_id"] == gold_custom_id else 0.0
                                for gold_custom_id in gold_custom_document_ids
                            ]
                        )

                    df_docs["gold_answers_match"] = df_docs.map_rows(
                        lambda row: [
                            1.0 if gold_answer != "" and gold_answer in row["context"] else 0.0
                            for gold_answer in gold_answers
                        ]
                    )

                    # document_relevance_criterion: "document_id"
                    df_docs["gold_id_match"] = df_docs.map_rows(lambda row: max(row["gold_documents_id_match"] + [0.0]))

                    # document_relevance_criterion: "answer",
                    df_docs["answer_match"] = df_docs.map_rows(lambda row: max(row["gold_answers_match"] + [0.0]))

                    # document_relevance_criterion: "document_id_or_answer",
                    df_docs["gold_id_or_answer_match"] = df_docs.map_rows(
                        lambda row: max(row["gold_id_match"], row["answer_match"])
                    )

                    # document_relevance_criterion: "document_id_and_answer",
                    df_docs["gold_id_and_answer_match"] = df_docs.map_rows(
                        lambda row: min(row["gold_id_match"], row["answer_match"])
                    )

                    # document_relevance_criterion: "context",
                    df_docs["context_match"] = df_docs.map_rows(
                        lambda row: 1.0
                        if any(sim for sim in row["gold_contexts_similarity"] if sim > context_matching_threshold)
                        else 0.0
                    )

                    # document_relevance_criterion: "document_id_or_context",
                    df_docs["gold_id_or_context_match"] = df_docs.map_rows(
                        lambda row: max(row["gold_id_match"], row["context_match"])
                    )

                    # document_relevance_criterion: "document_id_and_context",
                    df_docs["gold_id_and_context_match"] = df_docs.map_rows(
                        lambda row: min(row["gold_id_match"], row["context_match"])
                    )

                    # document_relevance_criterion: "document_id_and_context_and_answer",
                    df_docs["gold_id_and_context_and_answer_match"] = df_docs.map_rows(
                        lambda row: min(row["gold_id_match"], row["context_match"], row["answer_match"])
                    )

                    # document_relevance_criterion: "context_and_answer",
                    df_docs["context_and_answer_match"] = df_docs.map_rows(
                        lambda row: min(row["context_match"], row["answer_match"])
                    )

                    df_docs["rank"] = np.arange(1, len(df_docs) + 1)

                # add general info
                df_docs["type"] = "document"
                df_docs["node"] = node_name
                df_docs["multilabel_id"] = query_labels.id
                df_docs["query"] = query
                df_docs["filters"] = json.dumps(query_labels.filters, sort_keys=True).encode()
                df_docs["eval_mode"] = "isolated" if "isolated" in field_name else "integrated"
                partial_dfs.append(df_docs)

        if len(partial_dfs) == 0:
            return pd.DataFrame()

        return pd.concat(partial_dfs, ignore_index=True).reset_index()

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

        ``` python
        from haystack.document_stores.base import BaseDocumentStore
        INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
        res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)
        ```

        :return: List of components that are an instance the requested class
        """

        matches = [
            self.graph.nodes.get(node)["component"]
            for node in self.graph.nodes
            if isinstance(self.graph.nodes.get(node)["component"], class_type)
        ]
        return matches

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        """
        Return the document store object used in the current pipeline.

        :return: Instance of DocumentStore or None
        """
        matches = self.get_nodes_by_class(class_type=BaseDocumentStore)
        if len(matches) == 0:
            matches = list(
                set(retriever.document_store for retriever in self.get_nodes_by_class(class_type=BaseRetriever))
            )

        if len(matches) > 1:
            raise Exception(f"Multiple Document Stores found in Pipeline: {matches}")
        if len(matches) == 0:
            return None
        else:
            return matches[0]

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        try:
            import pygraphviz  # pylint: disable=unused-import
        except ImportError:
            raise ImportError(
                f"Could not import `pygraphviz`. Please install via: \n"
                f"pip install pygraphviz\n"
                f"(You might need to run this first: apt install libgraphviz-dev graphviz )"
            )

        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw(path)

    @classmethod
    def load_from_yaml(
        cls,
        path: Path,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

           ```yaml
           version: '1.9.0'

            components:    # define all the building-blocks for Pipeline
            - name: MyReader       # custom-name for the component; helpful for visualization & debugging
              type: FARMReader    # Haystack Class name for the component
              params:
                model_name_or_path: deepset/roberta-base-squad2
            - name: MyRetriever
              type: BM25Retriever
              params:
                document_store: MyDocumentStore    # params can reference other components defined in the YAML
            - name: MyDocumentStore
              type: ElasticsearchDocumentStore
              params:
                index: haystack_test

            pipelines:    # multiple Pipelines can be defined using the components from above
            - name: my_query_pipeline    # a simple extractive-qa Pipeline
              nodes:
              - name: MyRetriever
                inputs: [Query]
              - name: MyReader
                inputs: [MyRetriever]
           ```

        Note that, in case of a mismatch in version between Haystack and the YAML, a warning will be printed.
        If the pipeline loads correctly regardless, save again the pipeline using `Pipeline.save_to_yaml()` to remove the warning.

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        :param strict_version_check: whether to fail in case of a version mismatch (throws a warning otherwise)
        """

        config = read_pipeline_config_from_yaml(path)
        return cls.load_from_config(
            pipeline_config=config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
            strict_version_check=strict_version_check,
        )

    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
    ):
        """
        Load Pipeline from a config dict defining the individual components and how they're tied together to form
        a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

           ```python
           {
               "version": "ignore",
               "components": [
                   {  # define all the building-blocks for Pipeline
                       "name": "MyReader",  # custom-name for the component; helpful for visualization & debugging
                       "type": "FARMReader",  # Haystack Class name for the component
                       "params": {"no_ans_boost": -10, "model_name_or_path": "deepset/roberta-base-squad2"},
                   },
                   {
                       "name": "MyRetriever",
                       "type": "BM25Retriever",
                       "params": {
                           "document_store": "MyDocumentStore",  # params can reference other components defined in the YAML
                           "custom_query": None,
                       },
                   },
                   {"name": "MyDocumentStore", "type": "ElasticsearchDocumentStore", "params": {"index": "haystack_test"}},
               ],
               "pipelines": [
                   {  # multiple Pipelines can be defined using the components from above
                       "name": "my_query_pipeline",  # a simple extractive-qa Pipeline
                       "nodes": [
                           {"name": "MyRetriever", "inputs": ["Query"]},
                           {"name": "MyReader", "inputs": ["MyRetriever"]},
                       ],
                   }
               ],
           }
           ```

        :param pipeline_config: the pipeline config as dict
        :param pipeline_name: if the config contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        :param strict_version_check: whether to fail in case of a version mismatch (throws a warning otherwise).
        """
        validate_config(pipeline_config, strict_version_check=strict_version_check)
        pipeline = cls()

        pipeline_definition = get_pipeline_definition(pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config, overwrite_with_env_variables=overwrite_with_env_variables
        )
        components: Dict[str, BaseComponent] = {}
        for node_config in pipeline_definition["nodes"]:
            component = cls._load_or_get_component(
                name=node_config["name"], definitions=component_definitions, components=components
            )
            pipeline.add_node(component=component, name=node_config["name"], inputs=node_config["inputs"])

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
            logger.debug(f"Loading component '%s' of type '%s'", name, definitions[name]["type"])

            for key, value in component_params.items():
                # Component params can reference to other components. For instance, a Retriever can reference a
                # DocumentStore defined in the YAML. All references should be recursively resolved.
                if (
                    isinstance(value, str) and value in definitions.keys()
                ):  # check if the param value is a reference to another component.
                    if value not in components.keys():  # check if the referenced component is already loaded.
                        cls._load_or_get_component(name=value, definitions=definitions, components=components)
                    component_params[key] = components[
                        value
                    ]  # substitute reference (string) with the component object.

            component_instance = BaseComponent._create_instance(
                component_type=component_type, component_params=component_params, name=name
            )
            components[name] = component_instance
            return component_instance

        except KeyError as ke:
            raise PipelineConfigError(
                f"Failed loading pipeline component '{name}': "
                "seems like the component does not exist. Did you spell its name correctly?"
            ) from ke
        except Exception as e:
            raise PipelineConfigError(
                f"Failed loading pipeline component '{name}'. " "See the stacktrace above for more informations."
            ) from e

    def save_to_yaml(self, path: Path, return_defaults: bool = False):
        """
        Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

        :param path: path of the output YAML file.
        :param return_defaults: whether to output parameters that have the default values.
        """
        config = self.get_config(return_defaults=return_defaults)
        with open(path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def get_config(self, return_defaults: bool = False) -> dict:
        """
        Returns a configuration for the Pipeline that can be used with `Pipeline.load_from_config()`.

        :param return_defaults: whether to output parameters that have the default values.
        """
        if self.root_node:
            pipeline_name = ROOT_NODE_TO_PIPELINE_NAME[self.root_node.lower()]
        else:
            pipeline_name = "pipeline"

        pipeline_definitions: Dict[str, Dict] = {pipeline_name: {"name": pipeline_name, "nodes": []}}

        component_definitions: Dict[str, Dict] = {}
        for node_name, node_attributes in self.graph.nodes.items():
            if node_name == self.root_node:
                continue

            component: BaseComponent = node_attributes["component"]
            if node_name != component.name:
                raise PipelineError(f"Component name '{component.name}' does not match node name '{node_name}'.")

            self._add_component_to_definitions(
                component=component, component_definitions=component_definitions, return_defaults=return_defaults
            )

            # create the Pipeline definition with how the Component are connected
            inputs = []
            for predecessor in self.graph.predecessors(node_name):
                predecessor_out_edges = list(self.graph.edges(predecessor, data=True))
                # if there are multiple outputs and we're not coming from a root node we have to specify the stream
                if predecessor not in VALID_ROOT_NODES and len(predecessor_out_edges) > 1:
                    target_edge = next(edge for edge in predecessor_out_edges if edge[1] == node_name)
                    # data consists of a dictionary containing the stream_id at the "label" key
                    stream_id = target_edge[2]["label"]
                    inputs.append(f"{predecessor}.{stream_id}")
                else:
                    inputs.append(predecessor)

            pipeline_definitions[pipeline_name]["nodes"].append({"name": node_name, "inputs": inputs})

        config = {
            "components": list(component_definitions.values()),
            "pipelines": list(pipeline_definitions.values()),
            "version": __version__,
        }
        return config

    def _add_component_to_definitions(
        self, component: BaseComponent, component_definitions: Dict[str, Dict], return_defaults: bool = False
    ):
        """
        Add the definition of the component and all its dependencies (components too) to the component_definitions dict.
        This is used to collect all component definitions within Pipeline.get_config()
        """
        if component.name is None:
            raise PipelineError(f"Component with config '{component._component_config}' does not have a name.")

        component_params: Dict[str, Any] = component.get_params(return_defaults)
        # handling of subcomponents: add to definitions and substitute by reference
        for param_key, param_value in component_params.items():
            if isinstance(param_value, BaseComponent):
                sub_component = param_value
                self._add_component_to_definitions(sub_component, component_definitions, return_defaults)
                component_params[param_key] = sub_component.name

        component_definitions[component.name] = {
            "name": component.name,
            "type": component.type,
            "params": component_params,
        }

    def _get_all_component_names(self, components_to_search: Optional[List[BaseComponent]] = None) -> Set[str]:
        component_names = set()
        if components_to_search is None:
            components_to_search = list(self.components.values())
        for component in components_to_search:
            if component and component.name is not None:
                component_names.add(component.name)
                sub_component_names = self._get_all_component_names(component.utilized_components)
                component_names.update(sub_component_names)
        return component_names

    def _set_sub_component_names(self, component: BaseComponent, component_names: Set[str]):
        for sub_component in component.utilized_components:
            if sub_component.name is None:
                sub_component.name = self._generate_component_name(
                    type_name=sub_component.type, existing_component_names=component_names
                )
                component_names.add(sub_component.name)
            self._set_sub_component_names(sub_component, component_names=component_names)

    def _generate_component_name(self, type_name: str, existing_component_names: Set[str]) -> str:
        component_name: str = type_name
        # add number if there are multiple distinct ones of the same type
        while component_name in existing_component_names:
            occupied_num = 1
            if len(component_name) > len(type_name):
                occupied_num = int(component_name[len(type_name) + 1 :])
            new_num = occupied_num + 1
            component_name = f"{type_name}_{new_num}"
        return component_name

    def _validate_node_names_in_params(self, params: Optional[Dict]):
        """
        Validates the node names provided in the 'params' arg of run/run_batch method.
        """
        if params:
            if not all(node_id in self.graph.nodes for node_id in params.keys()):

                # Might be a non-targeted param. Verify that too
                not_a_node = set(params.keys()) - set(self.graph.nodes)
                # "debug" will be picked up by _dispatch_run, see its code
                # "add_isolated_node_eval" is set by pipeline.eval / pipeline.eval_batch
                valid_global_params = set(["debug", "add_isolated_node_eval"])
                for node_id in self.graph.nodes:
                    run_signature_args = self._get_run_node_signature(node_id)
                    valid_global_params |= set(run_signature_args)
                invalid_keys = [key for key in not_a_node if key not in valid_global_params]

                if invalid_keys:
                    raise ValueError(
                        f"No node(s) or global parameter(s) named {', '.join(invalid_keys)} found in pipeline."
                    )

    def _get_run_node_signature(self, node_id: str):
        return inspect.signature(self.graph.nodes[node_id]["component"].run).parameters.keys()

    def print_eval_report(
        self,
        eval_result: EvaluationResult,
        n_wrong_examples: int = 3,
        metrics_filter: Optional[Dict[str, List[str]]] = None,
        document_scope: Literal[
            "document_id",
            "context",
            "document_id_and_context",
            "document_id_or_context",
            "answer",
            "document_id_or_answer",
        ] = "document_id_or_answer",
        answer_scope: Literal["any", "context", "document_id", "document_id_and_context"] = "any",
        wrong_examples_fields: List[str] = ["answer", "context", "document_id"],
        max_characters_per_field: int = 150,
    ):
        """
        Prints evaluation report containing a metrics funnel and worst queries for further analysis.

        :param eval_result: The evaluation result, can be obtained by running eval().
        :param n_wrong_examples: The number of worst queries to show.
        :param metrics_filter: The metrics to show per node. If None all metrics will be shown.
        :param document_scope: A criterion for deciding whether documents are relevant or not.
            You can select between:
            - 'document_id': Specifies that the document ID must match. You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
                    A typical use case is Document Retrieval.
            - 'context': Specifies that the content of the document must match. Uses fuzzy matching (see `pipeline.eval()`'s `context_matching_...` params).
                    A typical use case is Document-Independent Passage Retrieval.
            - 'document_id_and_context': A Boolean operation specifying that both `'document_id' AND 'context'` must match.
                    A typical use case is Document-Specific Passage Retrieval.
            - 'document_id_or_context': A Boolean operation specifying that either `'document_id' OR 'context'` must match.
                    A typical use case is Document Retrieval having sparse context labels.
            - 'answer': Specifies that the document contents must include the answer. The selected `answer_scope` is enforced automatically.
                    A typical use case is Question Answering.
            - 'document_id_or_answer' (default): A Boolean operation specifying that either `'document_id' OR 'answer'` must match.
                    This is intended to be a proper default value in order to support both main use cases:
                    - Document Retrieval
                    - Question Answering
            The default value is 'document_id_or_answer'.
        :param answer_scope: Specifies the scope in which a matching answer is considered correct.
            You can select between:
            - 'any' (default): Any matching answer is considered correct.
            - 'context': The answer is only considered correct if its context matches as well.
                    Uses fuzzy matching (see `pipeline.eval()`'s `context_matching_...` params).
            - 'document_id': The answer is only considered correct if its document ID matches as well.
                    You can specify a custom document ID through `pipeline.eval()`'s `custom_document_id_field` param.
            - 'document_id_and_context': The answer is only considered correct if its document ID and its context match as well.
            The default value is 'any'.
            In Question Answering, to enforce that the retrieved document is considered correct whenever the answer is correct, set `document_scope` to 'answer' or 'document_id_or_answer'.
         :param wrong_examples_fields: A list of fields to include in the worst samples.
         :param max_characters_per_field: The maximum number of characters to include in the worst samples report (per field).
        """
        graph = DiGraph(self.graph.edges)
        print_eval_report(
            eval_result=eval_result,
            graph=graph,
            n_wrong_examples=n_wrong_examples,
            metrics_filter=metrics_filter,
            document_scope=document_scope,
            answer_scope=answer_scope,
            wrong_examples_fields=wrong_examples_fields,
            max_characters_per_field=max_characters_per_field,
        )

    def get_type(self) -> str:
        """
        Returns the type of the pipeline.
        """
        # values of the dict are functions evaluating whether components of this pipeline match the pipeline type
        # specified by dict keys
        pipeline_types = {
            # QuestionGenerationPipeline has only one component, which is a QuestionGenerator
            "QuestionGenerationPipeline": lambda x: all(isinstance(x, QuestionGenerator) for x in x.values()),
            # GenerativeQAPipeline has at least BaseGenerator and BaseRetriever components
            "GenerativeQAPipeline": lambda x: any(isinstance(x, BaseRetriever) for x in x.values())
            and any(isinstance(x, BaseGenerator) for x in x.values()),
            # FAQPipeline has at least one Docs2Answers component
            "FAQPipeline": lambda x: any(isinstance(x, Docs2Answers) for x in x.values()),
            # ExtractiveQAPipeline has at least one BaseRetriever component and one BaseReader component
            "ExtractiveQAPipeline": lambda x: any(isinstance(x, BaseRetriever) for x in x.values())
            and any(isinstance(x, BaseReader) for x in x.values()),
            # ExtractiveQAPipeline has at least one BaseSummarizer component and one BaseRetriever component
            "SearchSummarizationPipeline": lambda x: any(isinstance(x, BaseRetriever) for x in x.values())
            and any(isinstance(x, BaseSummarizer) for x in x.values()),
            # TranslationWrapperPipeline has two or more BaseTranslator components
            "TranslationWrapperPipeline": lambda x: [isinstance(x, BaseTranslator) for x in x.values()].count(True)
            >= 2,
            # RetrieverQuestionGenerationPipeline has at least one BaseRetriever component and one
            # QuestionGenerator component
            "RetrieverQuestionGenerationPipeline": lambda x: any(isinstance(x, BaseRetriever) for x in x.values())
            and any(isinstance(x, QuestionGenerator) for x in x.values()),
            # QuestionAnswerGenerationPipeline has at least one BaseReader component and one QuestionGenerator component
            "QuestionAnswerGenerationPipeline": lambda x: any(isinstance(x, BaseReader) for x in x.values())
            and any(isinstance(x, QuestionGenerator) for x in x.values()),
            # MostSimilarDocumentsPipeline has only BaseDocumentStore component
            "MostSimilarDocumentsPipeline": lambda x: len(x.values()) == 1
            and isinstance(list(x.values())[0], BaseDocumentStore),
            # DocumentSearchPipeline has at least one BaseRetriever component
            "DocumentSearchPipeline": lambda x: any(isinstance(x, BaseRetriever) for x in x.values()),
        }
        retrievers = [type(comp).__name__ for comp in self.components.values() if isinstance(comp, BaseRetriever)]
        doc_stores = [type(comp).__name__ for comp in self.components.values() if isinstance(comp, BaseDocumentStore)]

        pipeline_type = next(
            (p_type for p_type, eval_f in pipeline_types.items() if eval_f(self.components)), "Unknown pipeline"
        )
        retrievers_used = retrievers if retrievers else "None"
        doc_stores_used = doc_stores if doc_stores else "None"
        return f"{pipeline_type} (retriever: {retrievers_used}, doc_store: {doc_stores_used})"

    def uptime(self) -> timedelta:
        """
        Returns the uptime of the pipeline in timedelta.
        """
        return datetime.datetime.now(datetime.timezone.utc) - self.init_time

    def send_pipeline_event(self, is_indexing: bool = False):
        fingerprint = sha1(json.dumps(self.get_config(), sort_keys=True).encode()).hexdigest()
        send_custom_event(
            "pipeline",
            payload={
                "fingerprint": fingerprint,
                "type": "Indexing" if is_indexing else self.get_type(),
                "uptime": int(self.uptime().total_seconds()),
                "run_total": self.run_total,
                "run_total_window": self.run_total - self.last_window_run_total,
            },
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        self.time_of_last_sent_event = datetime.datetime(now.year, now.month, now.day, tzinfo=datetime.timezone.utc)
        self.last_window_run_total = self.run_total

    def send_pipeline_event_if_needed(self, is_indexing: bool = False):
        should_send_event = self.has_event_time_interval_exceeded() or self.has_event_run_total_threshold_exceeded()
        if should_send_event and not self.sent_event_in_window:
            self.send_pipeline_event(is_indexing)
            self.sent_event_in_window = True
        elif self.has_event_time_interval_exceeded():
            self.sent_event_in_window = False

    def has_event_time_interval_exceeded(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        return now - self.time_of_last_sent_event > self.event_time_interval

    def has_event_run_total_threshold_exceeded(self):
        return self.run_total - self.last_window_run_total > self.event_run_total_threshold


class _HaystackBeirRetrieverAdapter:
    def __init__(self, index_pipeline: Pipeline, query_pipeline: Pipeline, index_params: dict, query_params: dict):
        """
        Adapter mimicking a BEIR retriever used by BEIR's EvaluateRetrieval class to run BEIR evaluations on Haystack Pipelines.
        This has nothing to do with Haystack's retriever classes.
        See https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py.

        :param index_pipeline: The indexing pipeline to use.
        :param query_pipeline: The query pipeline to evaluate.
        :param index_params: The params to use during indexing (see pipeline.run's params).
        :param query_params: The params to use during querying (see pipeline.run's params).
        """
        self.index_pipeline = index_pipeline
        self.query_pipeline = query_pipeline
        self.index_params = index_params
        self.query_params = query_params

    def search(
        self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, score_function: str, **kwargs
    ) -> Dict[str, Dict[str, float]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            metas = []
            for id, doc in corpus.items():
                file_path = f"{temp_dir}/{id}"
                with open(file_path, "w") as f:
                    f.write(doc["text"])
                file_paths.append(file_path)
                metas.append({"id": id, "name": doc.get("title", None)})

            logger.info("indexing %s documents...", len(corpus))
            self.index_pipeline.run(file_paths=file_paths, meta=metas, params=self.index_params)
            logger.info("indexing finished.")

            # adjust query_params to ensure top_k is retrieved
            query_params = copy.deepcopy(self.query_params)
            query_params["top_k"] = top_k

            results = {}
            for q_id, query in tqdm(queries.items(), total=len(queries)):
                res = self.query_pipeline.run(query=query, params=query_params)
                docs = res["documents"]
                query_results = {doc.meta["id"]: doc.score for doc in docs}
                results[q_id] = query_results

            return results
