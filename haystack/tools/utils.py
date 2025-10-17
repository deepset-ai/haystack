# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset


def warm_up_tools(tools: Optional[Union[list[Tool], Toolset]] = None) -> None:
    """
    Warm up tools or a toolset.

    :param tools: A list of Tool objects or a Toolset to warm up.
    """
    if tools is None:
        return

    if isinstance(tools, list):
        # If tools is a list, warm up each tool individually
        for tool in tools:
            if hasattr(tool, "warm_up"):
                tool.warm_up()
    # If tools is a Toolset, warm up the toolset
    elif hasattr(tools, "warm_up"):
        tools.warm_up()
