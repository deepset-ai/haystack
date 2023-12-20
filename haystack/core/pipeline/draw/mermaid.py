# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import base64

import requests
import networkx  # type:ignore

from haystack.core.errors import PipelineDrawingError
from haystack.core.type_utils import _type_name

logger = logging.getLogger(__name__)


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

    connections_list = [
        f"{states[from_comp]} -- \"{conn_data['label']}<br><small><i>{conn_data['conn_type']}</i></small>\" --> {states[to_comp]}"
        for from_comp, to_comp, conn_data in graph.edges(data=True)
        if from_comp != "input" and to_comp != "output"
    ]
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
