# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional, Dict, get_args, Any

import logging
from pathlib import Path

import networkx

from canals.pipeline.validation import _find_pipeline_inputs, _find_pipeline_outputs
from canals.pipeline.draw.graphviz import _to_agraph
from canals.pipeline.draw.mermaid import _to_mermaid_image, _to_mermaid_text
from canals.type_utils import _type_name

logger = logging.getLogger(__name__)
RenderingEngines = Literal["graphviz", "mermaid-image", "mermaid-text"]


def _draw(
    graph: networkx.MultiDiGraph,
    path: Path,
    engine: RenderingEngines = "mermaid-image",
    style_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Renders the pipeline graph and saves it to file.
    """
    converted_graph = _convert(graph=graph, engine=engine, style_map=style_map)

    if engine == "graphviz":
        converted_graph.draw(path)

    elif engine == "mermaid-image":
        with open(path, "wb") as imagefile:
            imagefile.write(converted_graph)

    elif engine == "mermaid-text":
        with open((path), "w", encoding="utf-8") as textfile:
            textfile.write(converted_graph)

    else:
        raise ValueError(f"Unknown rendering engine '{engine}'. Choose one from: {get_args(RenderingEngines)}.")

    logger.debug("Pipeline diagram saved at %s", path)


def _convert_for_debug(
    graph: networkx.MultiDiGraph,
) -> Any:
    """
    Renders the pipeline graph with additional debug information into a text file that Mermaid can later render.
    """
    graph = _prepare_for_drawing(graph=graph, style_map={})
    return _to_mermaid_text(graph=graph)


def _convert(
    graph: networkx.MultiDiGraph,
    engine: RenderingEngines = "mermaid-image",
    style_map: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Renders the pipeline graph with the correct render and returns it.
    """
    graph = _prepare_for_drawing(graph=graph, style_map=style_map or {})

    if engine == "graphviz":
        return _to_agraph(graph=graph)

    if engine == "mermaid-image":
        return _to_mermaid_image(graph=graph)

    if engine == "mermaid-text":
        return _to_mermaid_text(graph=graph)

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
    for node, in_sockets in _find_pipeline_inputs(graph).items():
        for in_socket in in_sockets:
            if not in_socket.sender and not in_socket.is_optional:
                # If this socket has no sender it could be a socket that receives input
                # directly when running the Pipeline. We can't know that for sure, in doubt
                # we draw it as receiving input directly.
                graph.add_edge("input", node, label=in_socket.name, conn_type=_type_name(in_socket.type))

    # Draw the outputs
    graph.add_node("output")
    for node, out_sockets in _find_pipeline_outputs(graph).items():
        for out_socket in out_sockets:
            graph.add_edge(node, "output", label=out_socket.name, conn_type=_type_name(out_socket.type))

    return graph
