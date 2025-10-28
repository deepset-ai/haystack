# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: we do not use LazyImporter here because it creates conflicts between the tool module and the tool decorator

# ruff: noqa: I001 (ignore import order as we need to import Tool before ComponentTool and PipelineTool)
from typing import Sequence, Union

from haystack.tools.from_function import create_tool_from_function, tool
from haystack.tools.tool import Tool, _check_duplicate_tool_names
from haystack.tools.toolset import Toolset
from haystack.tools.component_tool import ComponentTool
from haystack.tools.pipeline_tool import PipelineTool
from haystack.tools.serde_utils import deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.tools.utils import flatten_tools_or_toolsets, warm_up_tools

# Type alias for tools parameter - allows mixing Tools and Toolsets in a sequence
# Using Sequence (covariant) instead of list (invariant) to maintain backward compatibility
# This allows list[Tool] to be passed where Sequence[Tool | Toolset] is expected
ToolsType = Union[Sequence[Union[Tool, Toolset]], Toolset]

__all__ = [
    "_check_duplicate_tool_names",
    "ComponentTool",
    "create_tool_from_function",
    "deserialize_tools_or_toolset_inplace",
    "flatten_tools_or_toolsets",
    "PipelineTool",
    "serialize_tools_or_toolset",
    "Tool",
    "ToolsType",
    "Toolset",
    "tool",
    "warm_up_tools",
]
