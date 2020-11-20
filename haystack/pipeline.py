import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from haystack.reader.base import BaseReader
from haystack.retriever.base import BaseRetriever


class QueryNode:
    outgoing_edges = 1

    def run(self, **kwargs):
        return kwargs, 1


class Pipeline:
    def __init__(self):
        self.graph = DiGraph()
        self.root_node_id = "Query"
        self.graph.add_node("Query", component=QueryNode())

    def add_node(self, component, name, inputs):
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

    def _get_next_nodes(self, node_id, stream_id):
        current_node_edges = self.graph.edges(node_id, data=True)
        next_nodes = [
            next_node
            for _, next_node, data in current_node_edges
            if not stream_id or data["label"] == f"output_{stream_id}"
        ]
        return next_nodes

    def draw(self):
        try:
            import pygraphviz
        except ImportError:
            raise ImportError(f"Could not import `pygraphviz`. Please install via: \n"
                              f"pip install pygraphviz\n"
                              f"(You might need to run this first: apt install libgraphviz-dev graphviz )")

        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw("pipeline.png")


class ExtractiveQAPipeline:
    def __init__(self, reader: BaseReader, retriever: BaseRetriever):
        """
        Initialize a Pipeline for Extractive Question Answering.

        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    def run(self, question, top_k_retriever=5, top_k_reader=5):
        output = self.pipeline.run(question=question,
                                   top_k_retriever=top_k_retriever,
                                   top_k_reader=top_k_reader)
        return output


class DocumentSearchPipeline:
    def __init__(self, retriever):
        """
        Initialize a Pipeline for semantic document search.

        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

    def run(self, question, top_k_retriever=5):
        output = self.pipeline.run(question=question, top_k_retriever=top_k_retriever)
        document_dicts = [doc.to_dict() for doc in output["documents"]]
        output["documents"] = document_dicts
        return output
