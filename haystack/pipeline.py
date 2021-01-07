from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict

import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack.generator.base import BaseGenerator
from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever
from haystack.summarizer.base import BaseSummarizer


class Pipeline:
    """
    Pipeline brings together building blocks to build a complex search pipeline with Haystack & user-defined components.

    Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
    flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
    Reader from multiple Retrievers, or re-ranking of candidate documents.
    """

    def __init__(self):
        self.graph = DiGraph()
        self.root_node_id = "Query"
        self.graph.add_node("Query", component=QueryNode())

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
        self.graph.add_node(name, component=component)

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

    def get_node(self, name: str):
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        component = self.graph.nodes[name]["component"]
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.graph.nodes[name]["component"] = component

    def run(self, **kwargs):
        has_next_node = True
        current_node_id = self.root_node_id
        input_dict = kwargs
        output_dict = None

        while has_next_node:
            output_dict, stream_id = self.graph.nodes[current_node_id]["component"].run(**input_dict)
            input_dict = output_dict
            next_nodes = self._get_next_nodes(current_node_id, stream_id)

            if len(next_nodes) > 1:
                join_node_id = list(nx.neighbors(self.graph, next_nodes[0]))[0]
                if set(self.graph.predecessors(join_node_id)) != set(next_nodes):
                    raise NotImplementedError(
                        "The current pipeline does not support multiple levels of parallel nodes."
                    )
                inputs_for_join_node = {"inputs": []}
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

    def _get_next_nodes(self, node_id: str, stream_id: str):
        current_node_edges = self.graph.edges(node_id, data=True)
        next_nodes = [
            next_node
            for _, next_node, data in current_node_edges
            if not stream_id or data["label"] == stream_id
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


class BaseStandardPipeline:
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

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: int = 10):
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

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: int = 10, top_k_generator: int = 10):
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
        top_k_retriever: int = 10,
        generate_single_summary: bool = False,
        return_in_answer_format=False
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

    def run(self, query: str, filters: Optional[Dict] = None, top_k_retriever: int = 10):
        output = self.pipeline.run(query=query, filters=filters, top_k_retriever=top_k_retriever)
        documents = output["documents"]

        results: Dict = {"query": query, "answers": []}
        for doc in documents:
            # TODO proper calibratation of pseudo probabilities
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


class QueryNode:
    outgoing_edges = 1

    def run(self, **kwargs):
        return kwargs, "output_1"


class JoinDocuments:
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
            for input_from_node, _ in inputs:
                for doc in input_from_node["documents"]:
                    document_map[doc.id] = doc
        elif self.join_mode == "merge":
            document_map = {}
            if self.weights:
                weights = self.weights
            else:
                weights = [1/len(inputs)] * len(inputs)
            for (input_from_node, _), weight in zip(inputs, weights):
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
        output = {"query": inputs[0][0]["query"], "documents": documents}
        return output, "output_1"
