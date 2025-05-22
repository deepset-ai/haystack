# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import colorsys
import json
import random
import zlib
from typing import Any, Dict, List, Optional

import networkx  # type:ignore
import requests

from haystack import logging
from haystack.core.errors import PipelineDrawingError
from haystack.core.pipeline.descriptions import find_pipeline_inputs, find_pipeline_outputs
from haystack.core.type_utils import _type_name

logger = logging.getLogger(__name__)


def generate_color_variations(n: int, base_color: Optional[str] = "#3498DB", variation_range=0.4) -> List[str]:
    """
    Generate n different variations of a base color.

    :param n: Number of variations to generate
    :param base_color: Hex color code, default is a shade of blue (#3498DB)
    :param variation_range: Range for varying brightness and saturation (0-1)

    :returns:
        list: List of hex color codes representing variations of the base color
    """
    # convert hex to RGB
    base_color = base_color.lstrip("#")  # type:ignore
    r = int(base_color[0:2], 16) / 255.0
    g = int(base_color[2:4], 16) / 255.0
    b = int(base_color[4:6], 16) / 255.0

    # convert RGB to HSV (Hue, Saturation, Value)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    variations = []
    for _ in range(n):
        # vary saturation and brightness within the specified range
        new_s = max(0, min(1, s + random.uniform(-variation_range, variation_range)))
        new_v = max(0, min(1, v + random.uniform(-variation_range, variation_range)))

        # keep hue the same for color consistency
        new_h = h

        # Convert back to RGB and then to hex
        new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, new_s, new_v)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(new_r * 255), int(new_g * 255), int(new_b * 255))

        variations.append(hex_color)

    return variations


def _prepare_for_drawing(graph: networkx.MultiDiGraph) -> networkx.MultiDiGraph:
    """
    Add some extra nodes to show the inputs and outputs of the pipeline.

    Also adds labels to edges.
    """
    # Label the edges
    for inp, outp, key, data in graph.edges(keys=True, data=True):
        data["label"] = (
            f"{data['from_socket'].name} -> {data['to_socket'].name}{' (opt.)' if not data['mandatory'] else ''}"
        )
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
%%{{ init: {params} }}%%

graph TD;

{connections}

classDef component text-align:center;
{style_definitions}
"""


def _validate_mermaid_params(params: Dict[str, Any]) -> None:
    """
    Validates and sets default values for Mermaid parameters.

    :param params:
        Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details.
        Supported keys:
            - format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
            - type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
            - theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
            - bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
            - width: Width of the output image (integer).
            - height: Height of the output image (integer).
            - scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
            - fit: Whether to fit the diagram size to the page (PDF only, boolean).
            - paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
            - landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.

    :raises ValueError:
        If any parameter is invalid or does not match the expected format.
    """
    valid_img_types = {"jpeg", "png", "webp"}
    valid_themes = {"default", "neutral", "dark", "forest"}
    valid_formats = {"img", "svg", "pdf"}

    params.setdefault("format", "img")
    params.setdefault("type", "png")
    params.setdefault("theme", "neutral")

    if params["format"] not in valid_formats:
        raise ValueError(f"Invalid image format: {params['format']}. Valid options are: {valid_formats}.")

    if params["format"] == "img" and params["type"] not in valid_img_types:
        raise ValueError(f"Invalid image type: {params['type']}. Valid options are: {valid_img_types}.")

    if params["theme"] not in valid_themes:
        raise ValueError(f"Invalid theme: {params['theme']}. Valid options are: {valid_themes}.")

    if "width" in params and not isinstance(params["width"], int):
        raise ValueError("Width must be an integer.")
    if "height" in params and not isinstance(params["height"], int):
        raise ValueError("Height must be an integer.")

    if "scale" in params and not 1 <= params["scale"] <= 3:
        raise ValueError("Scale must be a number between 1 and 3.")
    if "scale" in params and not ("width" in params or "height" in params):
        raise ValueError("Scale is only allowed when width or height is set.")

    if "bgColor" in params and not isinstance(params["bgColor"], str):
        raise ValueError("Background color must be a string.")

    # PDF specific parameters
    if params["format"] == "pdf":
        if "fit" in params and not isinstance(params["fit"], bool):
            raise ValueError("Fit must be a boolean.")
        if "paper" in params and not isinstance(params["paper"], str):
            raise ValueError("Paper size must be a string (e.g., 'a4', 'a3').")
        if "landscape" in params and not isinstance(params["landscape"], bool):
            raise ValueError("Landscape must be a boolean.")
        if "fit" in params and ("paper" in params or "landscape" in params):
            logger.warning("`fit` overrides `paper` and `landscape` for PDFs. Ignoring `paper` and `landscape`.")


def _to_mermaid_image(
    graph: networkx.MultiDiGraph,
    server_url: str = "https://mermaid.ink",
    params: Optional[dict] = None,
    timeout: int = 30,
    super_component_mapping: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Renders a pipeline using a Mermaid server.

    :param graph:
        The graph to render as a Mermaid pipeline.
    :param server_url:
        Base URL of the Mermaid server (default: 'https://mermaid.ink').
    :param params:
        Dictionary of customization parameters. See `validate_mermaid_params` for valid keys.
    :param timeout:
        Timeout in seconds for the request to the Mermaid server.
    :returns:
        The image, SVG, or PDF data returned by the Mermaid server as bytes.
    :raises ValueError:
        If any parameter is invalid or does not match the expected format.
    :raises PipelineDrawingError:
        If there is an issue connecting to the Mermaid server or the server returns an error.
    """

    if params is None:
        params = {}

    _validate_mermaid_params(params)

    theme = params.get("theme")
    init_params = json.dumps({"theme": theme})

    # Copy the graph to avoid modifying the original
    graph_styled = _to_mermaid_text(graph.copy(), init_params, super_component_mapping)
    json_string = json.dumps({"code": graph_styled})

    # Compress the JSON string with zlib (RFC 1950)
    compressor = zlib.compressobj(level=9, wbits=15)
    compressed_data = compressor.compress(json_string.encode("utf-8")) + compressor.flush()
    compressed_url_safe_base64 = base64.urlsafe_b64encode(compressed_data).decode("utf-8").strip()

    # Determine the correct endpoint
    endpoint_format = params.get("format", "img")  # Default to /img endpoint
    if endpoint_format not in {"img", "svg", "pdf"}:
        raise ValueError(f"Invalid format: {endpoint_format}. Valid options are 'img', 'svg', or 'pdf'.")

    # Construct the URL without query parameters
    url = f"{server_url}/{endpoint_format}/pako:{compressed_url_safe_base64}"

    # Add query parameters adhering to mermaid.ink documentation
    query_params = []
    for key, value in params.items():
        if key not in {"theme", "format"}:  # Exclude theme (handled in init_params) and format (endpoint-specific)
            if value is True:
                query_params.append(f"{key}")
            else:
                query_params.append(f"{key}={value}")

    if query_params:
        url += "?" + "&".join(query_params)

    logger.debug("Rendering graph at {url}", url=url)
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code >= 400:
            logger.warning(
                "Failed to draw the pipeline: {server_url} returned status {status_code}",
                server_url=server_url,
                status_code=resp.status_code,
            )
            logger.info("Exact URL requested: {url}", url=url)
            logger.warning("No pipeline diagram will be saved.")
            resp.raise_for_status()

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Failed to draw the pipeline: could not connect to {server_url} ({error})", server_url=server_url, error=exc
        )
        logger.info("Exact URL requested: {url}", url=url)
        logger.warning("No pipeline diagram will be saved.")
        raise PipelineDrawingError(f"There was an issue with {server_url}, see the stacktrace for details.") from exc

    return resp.content


def _to_mermaid_text(
    graph: networkx.MultiDiGraph, init_params: str, super_component_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Converts a Networkx graph into Mermaid syntax.

    The output of this function can be used in the documentation with `mermaid` codeblocks and will be
    automatically rendered.

    :param graph: The graph to convert to Mermaid syntax
    :param init_params: Initialization parameters for Mermaid
    :param super_component_mapping: Mapping of component names to super component names
    """
    # Copy the graph to avoid modifying the original
    graph = _prepare_for_drawing(graph.copy())
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

    # Create node definitions
    states = {}
    super_component_components = super_component_mapping.keys() if super_component_mapping else {}

    # color variations for super components
    super_component_colors = {}
    if super_component_components:
        unique_super_components = set(super_component_mapping.values())  # type:ignore
        color_variations = generate_color_variations(n=len(unique_super_components))
        super_component_colors = dict(zip(unique_super_components, color_variations))

    # Generate style definitions for each super component
    style_definitions = []
    for super_comp, color in super_component_colors.items():
        style_definitions.append(f"classDef {super_comp} fill:{color},color:white;")

    for comp, data in graph.nodes(data=True):
        if comp in ["input", "output"]:
            continue

        # styling based on whether the component is a SuperComponent
        if comp in super_component_components:
            super_component_name = super_component_mapping[comp]  # type:ignore
            style = super_component_name
        else:
            style = "component"
        node_def = f'{comp}["<b>{comp}</b><br><small><i>{type(data["instance"]).__name__}{optional_inputs[comp]}</i></small>"]:::{style}'  # noqa: E501
        states[comp] = node_def

    connections_list = []
    for from_comp, to_comp, conn_data in graph.edges(data=True):
        if from_comp != "input" and to_comp != "output":
            arrowtail = ARROWTAIL_MANDATORY if conn_data["mandatory"] else ARROWTAIL_OPTIONAL
            arrowhead = ARROWHEAD_MANDATORY if conn_data["mandatory"] else ARROWHEAD_OPTIONAL
            label = f'"{conn_data["label"]}<br><small><i>{conn_data["conn_type"]}</i></small>"'
            conn_string = f"{states[from_comp]} {arrowtail} {label} {arrowhead} {states[to_comp]}"
            connections_list.append(conn_string)

    input_connections = [
        f'i{{&ast;}}--"{conn_data["label"]}<br><small><i>{conn_data["conn_type"]}</i></small>"--> {states[to_comp]}'
        for _, to_comp, conn_data in graph.out_edges("input", data=True)
    ]
    output_connections = [
        f'{states[from_comp]}--"{conn_data["label"]}<br><small><i>{conn_data["conn_type"]}</i></small>"--> o{{&ast;}}'
        for from_comp, _, conn_data in graph.in_edges("output", data=True)
    ]
    connections = "\n".join(connections_list + input_connections + output_connections)

    # Create legend
    legend_nodes = []
    if super_component_colors:
        legend_nodes.append("subgraph Legend")
        for super_comp, color in super_component_colors.items():
            legend_id = f"legend_{super_comp}"
            legend_nodes.append(f'{legend_id}["{super_comp}"]:::{super_comp}')
        legend_nodes.append("end")
        connections += "\n" + "\n".join(legend_nodes)

    # Add style definitions to the template
    graph_styled = MERMAID_STYLED_TEMPLATE.format(
        params=init_params, connections=connections, style_definitions="\n".join(style_definitions)
    )
    logger.debug("Mermaid diagram:\n{diagram}", diagram=graph_styled)

    return graph_styled
