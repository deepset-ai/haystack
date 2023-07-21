# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
import logging
import base64
import inspect

import requests
import networkx

from canals.errors import PipelineDrawingError
from canals.sockets import get_socket_type_desc


logger = logging.getLogger(__name__)


MERMAID_STYLED_TEMPLATE = """
%%{{ init: {{'theme': 'neutral' }} }}%%

stateDiagram

{states}

{notes}

{transitions}

classDef components text-align:center;
"""


def _to_mermaid_image(graph: networkx.MultiDiGraph):
    """
    Renders a pipeline using Mermaid (hosted version at 'https://mermaid.ink'). Requires Internet access.
    """
    states, notes, transitions = _to_mermaid_text(graph=graph)

    graph_styled = MERMAID_STYLED_TEMPLATE.format(states=states, notes=notes, transitions=transitions)
    logger.debug("Mermaid diagram:\n%s", graph_styled)

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


def _to_mermaid_text(graph: networkx.MultiDiGraph) -> Tuple[str, str, str]:
    """
    Converts a Networkx graph into Mermaid syntax. The output of this function can be used in the documentation
    with `mermaid` codeblocks and it will be automatically rendered.
    """
    states = "\n".join(
        [
            f"{comp}:::components: {comp}<br><small><i>{type(data['instance']).__name__}</i></small>"
            for comp, data in graph.nodes(data=True)
            if comp != "input" and comp != "output"
        ]
    )
    sockets = {
        comp: "\n".join(
            [
                f"{name} <small><i>{get_socket_type_desc(socket.type)}</i></small>"
                for name, socket in data.get("input_sockets", {}).items()
                if socket.default is not inspect.Parameter.empty
            ]
        )
        for comp, data in graph.nodes(data=True)
    }
    notes = "\n".join(
        [
            f"note left of {comp}\n\t{sockets[comp]}\nend note"
            for comp in graph.nodes
            if comp != "input" and comp != "output" and sockets[comp] != ""
        ]
    )

    connections_list = [
        f"{from_comp} --> {to_comp} : {conn_data['label']}  <small><i>({conn_data['conn_type']})</i></small>"
        for from_comp, to_comp, conn_data in graph.edges(data=True)
        if from_comp != "input" and to_comp != "output"
    ]
    input_connections = [
        f"[*] --> {to_comp} : {conn_data['label']}  <small><i>({conn_data['conn_type']})</i></small>"
        for _, to_comp, conn_data in graph.out_edges("input", data=True)
    ]
    output_connections = [
        f"{from_comp} --> [*]  : {conn_data['label']}  <small><i>({conn_data['conn_type']})</i></small>"
        for from_comp, _, conn_data in graph.in_edges("output", data=True)
    ]
    connections = "\n".join(connections_list + input_connections + output_connections)

    # mermaid_graph = f"graph TD;\n{connections}"
    return states, notes, connections
