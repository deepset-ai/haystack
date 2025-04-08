# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Union

from haystack.core.errors import DeserializationError
from haystack.core.serialization import import_class_by_name
from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset


def serialize_tools_or_toolset(
    tools: Union[Toolset, List[Tool], None],
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Serialize a Toolset or a list of Tools to a dictionary or a list of tool dictionaries.

    :param tools: A Toolset, a list of Tools, or None
    :returns: A dictionary, a list of tool dictionaries, or None if tools is None
    """
    if tools is None:
        return None
    if isinstance(tools, Toolset):
        return tools.to_dict()
    return [tool.to_dict() for tool in tools]


def deserialize_tools_or_toolset_inplace(data: Dict[str, Any], key: str = "tools"):
    """
    Deserialize a list of Tools or a Toolset in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the list of Tools or Toolset is stored.
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

        deserialized_tools = []
        for tool in serialized_tools:
            if not isinstance(tool, dict):
                raise TypeError(f"Serialized tool '{tool}' is not a dictionary")

            # different classes are allowed: Tool, ComponentTool, etc.
            tool_class = import_class_by_name(tool["type"])
            if not issubclass(tool_class, Tool):
                raise TypeError(f"Class '{tool_class}' is not a subclass of Tool")

            deserialized_tools.append(tool_class.from_dict(tool))

        data[key] = deserialized_tools
