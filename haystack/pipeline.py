from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph


class Pipeline:
    def __init__(self):
        self.graph = DiGraph()
        self.root_node_id = None

    def add_node(self, component, name, inputs):
        if self.root_node_id is None:
            self.root_node_id = name
        self.graph.add_node(name, component=component)
        if inputs:
            for i in inputs:
                if "." in i:
                    [input_node_name, input_edge_name] = i.split(".")
                else:
                    num_edges_from_i = len(self.graph.edges(i))
                    input_edge_name = f"output_{num_edges_from_i}"
                    input_node_name = i
                self.graph.add_edge(input_node_name, name, label=input_edge_name)

    def run(self, **kwargs):
        has_next_node = True
        current_node_id = self.root_node_id
        input_dict = kwargs
        output_dict = None

        while has_next_node:
            output_dict, stream_id = self.graph.nodes[current_node_id]["component"].run(**input_dict)
            input_dict = output_dict
            current_node_edges = self.graph.edges(current_node_id, data=True)
            next_nodes = [
                next_node for _, next_node, data in current_node_edges if data["label"] == f"output_{stream_id}"
            ]
            if len(next_nodes) > 1:
                raise NotImplementedError("The current pipeline does not support parallel nodes.")
            elif len(next_nodes) == 1:
                current_node_id = next_nodes[0]
            else:
                has_next_node = False

        return output_dict

    def draw(self):
        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw("pipeline.png")
