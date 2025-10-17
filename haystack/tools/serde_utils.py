# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Union

from haystack.core.errors import DeserializationError
from haystack.core.serialization import import_class_by_name
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset

if TYPE_CHECKING:
    from haystack.tools import ToolsType


def serialize_tools_or_toolset(tools: "Optional[ToolsType]") -> Union[dict[str, Any], list[dict[str, Any]], None]:
    """
    Serialize tools or toolsets to dictionaries.

    :param tools: A Toolset, a list of Tools and/or Toolsets, or None
    :returns: Serialized representation preserving Tool/Toolset boundaries when provided
    """
    if tools is None:
        return None
    if isinstance(tools, Toolset):
        return tools.to_dict()
    if isinstance(tools, list):
        serialized: list[dict[str, Any]] = []
        for item in tools:
            if isinstance(item, (Toolset, Tool)):
                serialized.append(item.to_dict())
            else:
                raise TypeError("Items in the tools list must be Tool or Toolset instances.")
        return serialized
    raise TypeError("tools must be Toolset, list[Union[Tool, Toolset]], or None")


def deserialize_tools_or_toolset_inplace(data: dict[str, Any], key: str = "tools") -> None:
    """
    Deserialize a list of Tools and/or Toolsets, or a single Toolset in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the list of Tools and/or Toolsets, or single Toolset is stored.
    """
    if key in data:
        serialized_tools = data[key]

        if serialized_tools is None:
            return

        # Check if it's a serialized Toolset (a dict with "type" and "data" keys)
        if isinstance(serialized_tools, dict) and all(k in serialized_tools for k in ["type", "data"]):
            toolset_class_name = serialized_tools.get("type")
            if not toolset_class_name:
                raise DeserializationError("The 'type' key is missing or None in the serialized toolset data")

            toolset_class = import_class_by_name(toolset_class_name)

            if not issubclass(toolset_class, Toolset):
                raise TypeError(f"Class '{toolset_class}' is not a subclass of Toolset")

            data[key] = toolset_class.from_dict(serialized_tools)
            return

        if not isinstance(serialized_tools, list):
            raise TypeError(f"The value of '{key}' is not a list or a dictionary")

        deserialized_tools: list[Union[Tool, Toolset]] = []
        for tool in serialized_tools:
            if not isinstance(tool, dict):
                raise TypeError(f"Serialized tool '{tool}' is not a dictionary")

            # different classes are allowed: Tool, ComponentTool, Toolset, etc.
            tool_class = import_class_by_name(tool["type"])
            if issubclass(tool_class, (Tool, Toolset)):
                deserialized_tools.append(tool_class.from_dict(tool))
            else:
                raise TypeError(f"Class '{tool_class}' is neither Tool nor Toolset")

        data[key] = deserialized_tools
