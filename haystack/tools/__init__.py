# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: we do not use LazyImporter here because it creates conflicts between the tool module and the tool decorator

# ruff: noqa: I001 (ignore import order as we need to import Tool before ComponentTool)
from .tool import Tool, _check_duplicate_tool_names
from .toolset import Toolset
from .from_function import create_tool_from_function, tool
from .serde_utils import serialize_tools_or_toolset, deserialize_tools_or_toolset_inplace
from .component_tool import ComponentTool

__all__ = [
    "Tool",
    "Toolset",
    "create_tool_from_function",
    "tool",
    "serialize_tools_or_toolset",
    "deserialize_tools_or_toolset_inplace",
    "ComponentTool",
    "_check_duplicate_tool_names",
]
