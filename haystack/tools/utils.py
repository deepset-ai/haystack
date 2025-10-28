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

    :param tools: A list of Tool and/or Toolset objects, a single Toolset, or None.
    """
    if tools is None:
        return

    # If tools is a single Toolset, warm up the toolset itself
    if isinstance(tools, Toolset):
        if hasattr(tools, "warm_up"):
            tools.warm_up()
        return

    # If tools is a sequence, warm up each item (Tool or Toolset)
    # Note: Toolset is already handled above, so this handles list, tuple, etc.
    for item in tools:
        if isinstance(item, (Toolset, Tool)) and hasattr(item, "warm_up"):
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

    # Handle any sequence type (list, tuple, etc.) but not Toolset (already handled above)
    flattened: list[Tool] = []
    for item in tools:
        if isinstance(item, Toolset):
            flattened.extend(list(item))
        elif isinstance(item, Tool):
            flattened.append(item)
        else:
            raise TypeError("Items in the tools sequence must be Tool or Toolset instances.")
    return flattened
