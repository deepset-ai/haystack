import logging

import networkx
from networkx.drawing.nx_agraph import to_agraph as nx_to_agraph

logger = logging.getLogger(__name__)

try:
    from pygraphviz import AGraph
except ImportError:
    logger.debug(
        "Could not import `pygraphviz`. Please install via: \n"
        "pip install pygraphviz\n"
        "(You might need to run this first: apt install libgraphviz-dev graphviz )"
    )
    AGraph = None


def to_agraph(graph: networkx.MultiDiGraph) -> AGraph:
    """
    Renders a pipeline graph using PyGraphViz. You need to install it and all its system dependencies for it to work.
    """
    if not AGraph:
        raise ImportError(
            "Could not import `pygraphviz`. Please install via: \n"
            "pip install pygraphviz\n"
            "(You might need to run this first: apt install libgraphviz-dev graphviz )"
        )

    for node in graph.nodes:
        if graph.nodes[node].get("style") == "running":
            graph.nodes[node]["color"] = "red"
            graph.nodes[node]["penwidth"] = 3

        if graph.nodes[node].get("style") == "queued":
            graph.nodes[node]["color"] = "#0f0"
            graph.nodes[node]["penwidth"] = 3

    for inp, outp, key, data in graph.out_edges("input", keys=True, data=True):
        data["style"] = "dashed"
        graph.add_edge(inp, outp, key=key, **data)

    for inp, outp, key, data in graph.in_edges("output", keys=True, data=True):
        data["style"] = "dashed"
        graph.add_edge(inp, outp, key=key, **data)

    graph.nodes["input"]["shape"] = "plain"
    graph.nodes["output"]["shape"] = "plain"
    agraph = nx_to_agraph(graph)
    agraph.layout("dot")
    return agraph
