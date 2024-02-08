# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import logging
from pathlib import Path
from typing import Optional

import networkx  # type:ignore
import requests

from haystack.core.errors import PipelineDrawingError
from haystack.core.pipeline.descriptions import find_pipeline_inputs, find_pipeline_outputs
from haystack.core.type_utils import _type_name

logger = logging.getLogger(__name__)


def _draw(graph: networkx.MultiDiGraph, path: Optional[Path] = None) -> None:
    """
    Draw a pipeline graph using Mermaid and save it to a file.
    If on a Jupyter notebook, it will also display the image inline.
    """
    image_data = _to_mermaid_image(_prepare_for_drawing(graph))

    in_notebook = False
    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import Image, display

        if "IPKernelApp" in get_ipython().config:
            # We're in a notebook, let's display the image
            display(Image(image_data))
            in_notebook = True
    except ImportError:
        pass
    except AttributeError:
        pass

    if not in_notebook and not path:
        # We're not in a notebook and no path is given, the user must have forgot
        # to specify the path. Raise an error.
        msg = "No path specified to save the image to."
        raise ValueError(msg)

    if path:
        # If we reached this point we're in a notebook and the user has specified a path.
        # Let's save the image anyway even if it's been displayed in the notebook.
        Path(path).write_bytes(image_data)


def _prepare_for_drawing(graph: networkx.MultiDiGraph) -> networkx.MultiDiGraph:
    """
    Add some extra nodes to show the inputs and outputs of the pipeline.
    Also adds labels to edges.
    """
    # Label the edges
    for inp, outp, key, data in graph.edges(keys=True, data=True):
        data[
            "label"
        ] = f"{data['from_socket'].name} -> {data['to_socket'].name}{' (opt.)' if not data['mandatory'] else ''}"
        graph.add_edge(inp, outp, key=key, **data)

    # Add inputs fake node
    graph.add_node("input")
    for node, in_sockets in find_pipeline_inputs(graph).items():
        for in_socket in in_sockets:
            if not in_socket.senders and in_socket.is_mandatory:
                # If this socket has no sender it could be a socket that receives input
                # directly when running the Pipeline. We can't know that for sure, in doubt
                # we draw it as receiving input directly.
                graph.add_edge("input", node, label=in_socket.name, conn_type=_type_name(in_socket.type))

    # Add outputs fake node
    graph.add_node("output")
    for node, out_sockets in find_pipeline_outputs(graph).items():
        for out_socket in out_sockets:
            graph.add_edge(node, "output", label=out_socket.name, conn_type=_type_name(out_socket.type))

    return graph


ARROWTAIL_MANDATORY = "--"
ARROWTAIL_OPTIONAL = "-."
ARROWHEAD_MANDATORY = "-->"
ARROWHEAD_OPTIONAL = ".->"
MERMAID_STYLED_TEMPLATE = """
%%{{ init: {{'theme': 'neutral' }} }}%%

graph TD;

{connections}

classDef component text-align:center;
"""


def _to_mermaid_image(graph: networkx.MultiDiGraph):
    """
    Renders a pipeline using Mermaid (hosted version at 'https://mermaid.ink'). Requires Internet access.
    """
    graph_styled = _to_mermaid_text(graph=graph)

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
            resp.raise_for_status()

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to draw the pipeline: could not connect to https://mermaid.ink/img/ (%s)", exc)
        logger.info("Exact URL requested: %s", url)
        logger.warning("No pipeline diagram will be saved.")
        raise PipelineDrawingError(
            "There was an issue with https://mermaid.ink/, see the stacktrace for details."
        ) from exc

    return resp.content


def _to_mermaid_text(graph: networkx.MultiDiGraph) -> str:
    """
    Converts a Networkx graph into Mermaid syntax. The output of this function can be used in the documentation
    with `mermaid` codeblocks and it will be automatically rendered.
    """
    sockets = {
        comp: "".join(
            [
                f"<li>{name} ({_type_name(socket.type)})</li>"
                for name, socket in data.get("input_sockets", {}).items()
                if (not socket.is_mandatory and not socket.senders) or socket.is_variadic
            ]
        )
        for comp, data in graph.nodes(data=True)
    }
    optional_inputs = {
        comp: f"<br><br>Optional inputs:<ul style='text-align:left;'>{sockets}</ul>" if sockets else ""
        for comp, sockets in sockets.items()
    }

    states = {
        comp: f"{comp}[\"<b>{comp}</b><br><small><i>{type(data['instance']).__name__}{optional_inputs[comp]}</i></small>\"]:::component"
        for comp, data in graph.nodes(data=True)
        if comp not in ["input", "output"]
    }

    connections_list = []
    for from_comp, to_comp, conn_data in graph.edges(data=True):
        if from_comp != "input" and to_comp != "output":
            arrowtail = ARROWTAIL_MANDATORY if conn_data["mandatory"] else ARROWTAIL_OPTIONAL
            arrowhead = ARROWHEAD_MANDATORY if conn_data["mandatory"] else ARROWHEAD_OPTIONAL
            label = f'"{conn_data["label"]}<br><small><i>{conn_data["conn_type"]}</i></small>"'
            conn_string = f"{states[from_comp]} {arrowtail} {label} {arrowhead} {states[to_comp]}"
            connections_list.append(conn_string)

    input_connections = [
        f"i{{*}} -- \"{conn_data['label']}<br><small><i>{conn_data['conn_type']}</i></small>\" --> {states[to_comp]}"
        for _, to_comp, conn_data in graph.out_edges("input", data=True)
    ]
    output_connections = [
        f"{states[from_comp]} -- \"{conn_data['label']}<br><small><i>{conn_data['conn_type']}</i></small>\"--> o{{*}}"
        for from_comp, _, conn_data in graph.in_edges("output", data=True)
    ]
    connections = "\n".join(connections_list + input_connections + output_connections)

    graph_styled = MERMAID_STYLED_TEMPLATE.format(connections=connections)
    logger.debug("Mermaid diagram:\n%s", graph_styled)

    return graph_styled
