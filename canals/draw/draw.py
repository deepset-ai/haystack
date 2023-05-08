from typing import Literal, Optional, Dict, get_args, Any

import logging
from pathlib import Path

import networkx

from canals.pipeline._utils import find_pipeline_inputs, find_pipeline_outputs
from canals.draw.graphviz import to_agraph
from canals.draw.mermaid import to_mermaid_image, to_mermaid_text


logger = logging.getLogger(__name__)
RenderingEngines = Literal["graphviz", "mermaid-img", "mermaid-text"]


def draw(
    graph: networkx.MultiDiGraph,
    path: Path,
    engine: RenderingEngines = "mermaid-img",
    style_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Renders the pipeline graph and saves it to file.
    """
    converted_graph = convert(graph=graph, engine=engine, style_map=style_map)

    if engine == "graphviz":
        converted_graph.draw(path)

    elif engine == "mermaid-img":
        with open(path, "wb") as imagefile:
            imagefile.write(converted_graph)

    elif engine == "mermaid-text":
        with open((path), "w", encoding="utf-8") as textfile:
            textfile.write(converted_graph)

    else:
        raise ValueError(f"Unknown rendering engine '{engine}'. Choose one from: {get_args(RenderingEngines)}.")

    logger.debug("Pipeline diagram saved at %s", path)


def convert(
    graph: networkx.MultiDiGraph,
    engine: RenderingEngines = "mermaid-img",
    style_map: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Renders the pipeline graph with the correct render and returns it.
    """
    graph = _prepare_for_drawing(graph=graph, style_map=style_map or {})

    if engine == "graphviz":
        return to_agraph(graph=graph)

    if engine == "mermaid-img":
        return to_mermaid_image(graph=graph)

    if engine == "mermaid-text":
        return to_mermaid_text(graph=graph)

    raise ValueError(f"Unknown rendering engine '{engine}'. Choose one from: {get_args(RenderingEngines)}.")


def _prepare_for_drawing(graph: networkx.MultiDiGraph, style_map: Dict[str, str]) -> networkx.MultiDiGraph:
    """
    Prepares the graph to be drawn: adds explitic input and output nodes, labels the edges, applies the styles, etc.
    """
    # Apply the styles
    if style_map:
        for node, style in style_map.items():
            graph.nodes[node]["style"] = style

    # Label the edges
    for inp, outp, key, data in graph.edges(keys=True, data=True):
        data["label"] = f"{data['from_socket'].name} -> {data['to_socket'].name}"
        graph.add_edge(inp, outp, key=key, **data)

    # Draw the inputs
    graph.add_node("input")
    for node, in_sockets in find_pipeline_inputs(graph).items():
        for in_socket in in_sockets:
            node_instance = graph.nodes[node]["instance"]
            input_node_defaults = hasattr(node_instance, "defaults") and in_socket.name in node_instance.defaults
            if not input_node_defaults and not (graph.nodes[node]["variadic_input"] and not graph.in_edges(node)):
                graph.add_edge("input", node, label=in_socket.name)

    # Draw the outputs
    graph.add_node("output")
    for node, out_sockets in find_pipeline_outputs(graph).items():
        for out_socket in out_sockets:
            graph.add_edge(node, "output", label=out_socket.name)

    return graph
