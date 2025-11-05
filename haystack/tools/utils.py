# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset

if TYPE_CHECKING:
    from haystack.tools import ToolsType


def warm_up_tools(tools: "Optional[ToolsType]" = None) -> None:
    """
    Warm up tools from various formats (Tools, Toolsets, or mixed lists).

    For Toolset objects, this delegates to Toolset.warm_up(), which by default
    warms up all tools in the Toolset. Toolset subclasses can override warm_up()
    to customize initialization behavior (e.g., setting up shared resources).

    :param tools: A list of Tool and/or Toolset objects, a single Toolset, or None.
    """
    if tools is None:
        return

    # If tools is a single Toolset or Tool, warm it up
    if isinstance(tools, (Toolset, Tool)):
        if hasattr(tools, "warm_up"):
            tools.warm_up()
        return

    # If tools is a list, warm up each item (Tool or Toolset)
    if isinstance(tools, list):
        for item in tools:
            if hasattr(item, "warm_up"):
                item.warm_up()


def flatten_tools_or_toolsets(tools: "Optional[ToolsType]") -> list[Tool]:
    """
    Flatten tools from various formats into a list of Tool instances.

    :param tools: Tools in list[Union[Tool, Toolset]], Toolset, or None format.
    :returns: A flat list of Tool instances.
    """
    if tools is None:
        return []

    if isinstance(tools, Toolset):
        return list(tools)

    if isinstance(tools, list):
        flattened: list[Tool] = []
        for item in tools:
            if isinstance(item, Toolset):
                flattened.extend(list(item))
            elif isinstance(item, Tool):
                flattened.append(item)
            else:
                raise TypeError("Items in the tools list must be Tool or Toolset instances.")
        return flattened

    raise TypeError("tools must be list[Union[Tool, Toolset]], Toolset, or None")
