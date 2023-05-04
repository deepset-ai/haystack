from typing import Literal, Union, Optional, List

import logging
import base64
from pathlib import Path

import requests
import networkx
from networkx.drawing.nx_agraph import to_agraph

from canals.pipeline._utils import find_pipeline_inputs, find_pipeline_outputs


logger = logging.getLogger(__name__)


def draw(  # pylint: disable=too-many-locals
    graph: networkx.MultiDiGraph,
    path: Path,
    engine: Literal["graphviz", "mermaid"] = "mermaid",
    running: Optional[str] = None,
    queued: Optional[List[str]] = None,
) -> None:
    """
    Prepares the graph for rendering with either graphviz or mermaid, then invokes the correct renderer.
    """
    # Mark the highlights, if any
    if running:
        graph.nodes[running]["running"] = True
    if queued:
        for node in queued:
            graph.nodes[node]["queued"] = True

    # Label the edges
    for inp, outp, key, data in graph.edges(keys=True, data=True):
        data["label"] = f"{data['from_socket'].name} -> {data['to_socket'].name}"
        graph.add_edge(inp, outp, key=key, **data)

    input_nodes = find_pipeline_inputs(graph)
    output_nodes = find_pipeline_outputs(graph)

    # Draw the inputs
    graph.add_node("input")
    for in_node, in_sockets in input_nodes.items():
        for in_socket in in_sockets:
            if not in_socket.has_default and not (
                graph.nodes[in_node]["variadic_input"] and not graph.in_edges(in_node)
            ):
                graph.add_edge("input", in_node, label=in_socket.name)

    # Draw the outputs
    graph.add_node("output")
    for out_node, out_sockets in output_nodes.items():
        for out_socket in out_sockets:
            graph.add_edge(out_node, "output", label=out_socket.name)

    if engine == "graphviz":
        _render_graphviz(graph=graph, path=path)
    elif engine == "mermaid":
        _render_mermaid(graph=graph, path=path)
    else:
        raise ValueError(f"Unknown rendering engine '{engine}'. Choose one from: 'graphviz', 'mermaid'.")

    logger.debug("Pipeline diagram saved at %s", path)


def _render_graphviz(graph: networkx.MultiDiGraph, path: Union[str, Path]):
    """
    Renders a pipeline graph using PyGraphViz. You need to install it and all its system dependencies for it to work.
    """
    try:
        import pygraphviz  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        logger.debug(
            "Could not import `pygraphviz`. Please install via: \n"
            "pip install pygraphviz\n"
            "(You might need to run this first: apt install libgraphviz-dev graphviz )"
        )
    for node in graph.nodes:
        if "running" in graph.nodes[node]:
            graph.nodes[node]["color"] = "red"
            graph.nodes[node]["penwidth"] = 3
        if "queued" in graph.nodes[node]:
            graph.nodes[node]["color"] = "#0f0"
            graph.nodes[node]["penwidth"] = 3

    graph.nodes["input"]["shape"] = "plain"
    graph.nodes["output"]["shape"] = "plain"
    graphviz = to_agraph(graph)
    graphviz.layout("dot")
    graphviz.draw(path)


def _render_mermaid(graph: networkx.MultiDiGraph, path: Union[str, Path]):
    """
    Renders a pipeline using Mermaid (hosted version at 'https://mermaid.ink'). Requires Internet access.
    """
    graph_as_text = to_mermaid(graph=graph)
    num_solid_arrows = len(graph.edges) - len(graph.in_edges("output")) - len(graph.out_edges("input"))
    solid_arrows = "\n".join([f"linkStyle {i} stroke-width:2px;" for i in range(num_solid_arrows)])
    graph_styled = f"""
%%{{ init: {{'theme': 'neutral' }} }}%%

{graph_as_text}

style IN  fill:#fff,stroke:#fff,stroke-width:1px
style OUT fill:#fff,stroke:#fff,stroke-width:1px
classDef running fill:#fff,stroke:#f00,stroke-width:2px
classDef queued fill:#fff,stroke:#0f0,stroke-width:2px
linkStyle default stroke-width:2px,stroke-dasharray: 5 5;
{solid_arrows}
"""
    logger.debug("Mermaid diagram:\n%s", graph_styled)

    # linkStyle default stroke:#999,stroke-width:2px,color:black;
    # classDef default fill:#fff,stroke:#000,stroke-width:1px;
    graphbytes = graph_styled.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = "https://mermaid.ink/img/" + base64_string

    logging.debug("Rendeding graph at %s", url)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code >= 400:
            logger.warning("Failed to draw the pipeline: https://mermaid.ink/img/ returned status %s", resp.status_code)
            logger.info("Exact URL requested: %s", url)
            logger.warning("No pipeline diagram will be saved.")
            return
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to draw the pipeline: could not connect to https://mermaid.ink/img/ (%s)", exc)
        logger.info("Exact URL requested: %s", url)
        logger.warning("No pipeline diagram will be saved.")
        return

    image = resp.content
    with open(path, "wb") as imagefile:
        imagefile.write(image)


def to_mermaid(graph: networkx.MultiDiGraph) -> str:
    """
    Converts a Networkx graph into Mermaid syntax. The output of this function can be used in the documentation
    with `mermaid` codeblocks and it will be automatically rendered.
    """
    components = {
        comp: f"{comp}:::{highlight}" if highlight in graph.nodes[comp] else comp
        for highlight in ["running", "queued"]
        for comp in graph.nodes
    }

    connections_list = [
        f"{components[from_comp]} -- {conn_data['label']} --> {components[to_comp]}"
        for from_comp, to_comp, conn_data in graph.edges(data=True)
        if from_comp != "input" and to_comp != "output"
    ]
    input_connections = [
        f"IN([input]) -- {conn_data['label']} --> {components[to_comp]}"
        for _, to_comp, conn_data in graph.out_edges("input", data=True)
    ]
    output_connections = [
        f"{components[from_comp]} -- {conn_data['label']} --> OUT([output])"
        for from_comp, _, conn_data in graph.in_edges("output", data=True)
    ]
    connections = "\n".join(connections_list + input_connections + output_connections)

    mermaid_graph = f"graph TD;\n{connections}"
    return mermaid_graph
