# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset


def flatten_tools_or_toolsets(tools: Optional[Union[list[Union[Tool, Toolset]], Toolset]]) -> list[Tool]:
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
