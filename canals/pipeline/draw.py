from typing import Literal, Union

import logging
import base64
from pathlib import Path

import requests
import networkx
from networkx.drawing.nx_agraph import to_agraph


logger = logging.getLogger(__name__)


def render_graphviz(graph: networkx.MultiDiGraph, path: Union[str, Path]):
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
    graph.nodes["input"]["shape"] = "plain"
    graph.nodes["output"]["shape"] = "plain"
    graphviz = to_agraph(graph)
    graphviz.layout("dot")
    graphviz.draw(path)


def render_mermaid(graph: networkx.MultiDiGraph, path: Union[str, Path]):
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


def to_mermaid(graph: networkx.MultiDiGraph, orientation: Literal["top-down", "left-right"] = "top-down") -> str:
    """
    Converts a Networkx graph into Mermaid syntax. The output of this function can be used in the documentation
    with `mermaid` codeblocks and it will be automatically rendered.
    """
    if orientation == "top-down":
        header = "graph TD;"
    elif orientation == "left-right":
        header = "graph LR;"
    else:
        raise ValueError(f"Unknown orientation '{orientation}': choose one of 'top-down', 'left-right'")

    connections_list = [
        f"{from_comp} -- {conn_data['label']} --> {to_comp}"
        for from_comp, to_comp, conn_data in graph.edges(data=True)
        if from_comp != "input" and to_comp != "output"
    ]
    input_connections = [
        f"IN([input]) -- {conn_data['label']} --> {to_comp}"
        for _, to_comp, conn_data in graph.out_edges("input", data=True)
    ]
    output_connections = [
        f"{from_comp} -- {conn_data['label']} --> OUT([output])"
        for from_comp, _, conn_data in graph.in_edges("output", data=True)
    ]
    connections = "\n".join(connections_list + input_connections + output_connections)

    mermaid_graph = f"{header}\n{connections}"
    return mermaid_graph
