# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset

if TYPE_CHECKING:
    from haystack.tools import ToolsType


def _as_tool_sequence(tools: "ToolsType | list[str] | None") -> Sequence[Tool | Toolset]:
    if tools is None:
        return []
    if isinstance(tools, Toolset):
        return [tools]
    return [item for item in tools if not isinstance(item, str)]


def warm_up_tools(tools: "ToolsType | list[str] | None" = None) -> None:
    """
    Warm up tools from various formats (Tools, Toolsets, or mixed lists).

    For Toolset objects, this delegates to warm_up() if implemented; otherwise,
    it calls warm_up() on each tool that implements it. Toolset subclasses can implement warm_up()
    to customize initialization behavior (e.g., setting up shared resources).

    :param tools: A sequence of Tool and/or Toolset objects, a single Toolset,
        a list of tool names, or None. Lists of names are ignored.
    """
    for item in _as_tool_sequence(tools):
        if hasattr(item, "warm_up"):
            item.warm_up()
        elif isinstance(item, Toolset):
            warm_up_tools(tools=item.tools)


async def warm_up_tools_async(tools: "ToolsType | list[str] | None" = None) -> None:
    """
    Warm up tools asynchronously from various formats (Tools, Toolsets, or mixed lists).

    Call warm_up_async() if implemented, otherwise warm_up() if available.
    If a Toolset implements neither method, this function warms up its tools.

    :param tools: A sequence of Tool and/or Toolset objects, a single Toolset,
        a list of tool names, or None. Lists of names are ignored.
    """
    for item in _as_tool_sequence(tools):
        if hasattr(item, "warm_up_async"):
            await item.warm_up_async()
        elif hasattr(item, "warm_up"):
            item.warm_up()
        elif isinstance(item, Toolset):
            await warm_up_tools_async(tools=item.tools)


def close_tools(tools: "ToolsType | list[str] | None" = None) -> None:
    """
    Close tools from various formats (Tools, Toolsets, or mixed lists).

    For Toolset objects, this delegates to close() if implemented; otherwise,
    it calls close() on each tool that implements it.

    :param tools: A sequence of Tool and/or Toolset objects, a single Toolset,
        a list of tool names, or None. Lists of names are ignored.
    """
    for item in _as_tool_sequence(tools):
        if hasattr(item, "close"):
            item.close()
        elif isinstance(item, Toolset):
            close_tools(tools=item.tools)


async def close_tools_async(tools: "ToolsType | list[str] | None" = None) -> None:
    """
    Close tools asynchronously from various formats (Tools, Toolsets, or mixed lists).

    Call close_async() if implemented, otherwise close() if available.
    If a Toolset implements neither method, this function closes its tools.

    :param tools: A sequence of Tool and/or Toolset objects, a single Toolset,
        a list of tool names, or None. Lists of names are ignored.
    """
    for item in _as_tool_sequence(tools):
        if hasattr(item, "close_async"):
            await item.close_async()
        elif hasattr(item, "close"):
            item.close()
        elif isinstance(item, Toolset):
            await close_tools_async(tools=item.tools)


def flatten_tools_or_toolsets(tools: "ToolsType | None") -> list[Tool]:
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
