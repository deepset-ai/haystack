# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset


def _ls_state(state: State) -> str:
    """List all available keys in the agent state and their types."""
    lines = []
    for key, definition in state.schema.items():
        type_name = getattr(definition["type"], "__name__", str(definition["type"]))
        lines.append(f"- {key} ({type_name})")
    return "\n".join(lines)


def _read_state(key: str, state: State, truncate: bool = True) -> str:
    """Read the value of a key from the agent state."""
    if not state.has(key):
        return f"Key '{key}' not found in state. Call ls_state to see available keys."
    value = repr(state.get(key))
    if truncate and len(value) > 200:
        value = value[:200] + "... [truncated, call cat_state with truncate=False to see full value]"
    return value


def _write_state(key: str, value: str, state: State) -> str:
    """Write a string value to a key in the agent state. Only string-typed keys are supported."""
    definition = state.schema.get(key)
    if definition is None:
        return f"Key '{key}' not found in state. Call ls_state to see available keys."
    if definition["type"] is not str:
        type_name = getattr(definition["type"], "__name__", str(definition["type"]))
        return f"Key '{key}' has type '{type_name}'. write_state only supports string-typed keys."
    state.set(key, value)
    return f"State key '{key}' updated successfully."


class LsStateTool(Tool):
    """
    A pre-built tool that lists all keys and their types in the agent state.

    The agent can call this as a cheap first step to discover what state is available
    before deciding whether to read any values.
    """

    def __init__(
        self,
        *,
        name: str = "ls_state",
        description: str = "List all available keys in the agent state and their types.",
    ) -> None:
        """
        Initialize the LsStateTool.

        :param name: Name of the tool.
        :param description: Description of the tool shown to the LLM.
        """
        super().__init__(
            name=name, description=description, parameters={"type": "object", "properties": {}}, function=_ls_state
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"name": self.name, "description": self.description},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LsStateTool":
        """
        Deserializes the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        return cls(**data["data"])


class ReadStateTool(Tool):
    """
    A pre-built tool that reads the value of a single key from the agent state.
    """

    def __init__(
        self,
        *,
        name: str = "read_state",
        description: str = "Read the value of a key from the agent state.",
        key_description: str = "The state key to read. Call ls_state to see available keys.",
        truncate_description: str = (
            "If True, the value is truncated to 200 characters. Set to False to see the full value."
        ),
    ) -> None:
        """
        Initialize the CatStateTool.

        :param name: Name of the tool.
        :param description: Description of the tool shown to the LLM.
        :param key_description: Description of the `key` parameter shown to the LLM.
        :param truncate_description: Description of the `truncate` parameter shown to the LLM.
        """
        self.key_description = key_description
        self.truncate_description = truncate_description

        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": key_description},
                    "truncate": {"type": "boolean", "description": truncate_description, "default": False},
                },
                "required": ["key"],
            },
            function=_read_state,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "name": self.name,
                "description": self.description,
                "key_description": self.key_description,
                "truncate_description": self.truncate_description,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadStateTool":
        """
        Deserializes the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        return cls(**data["data"])


class WriteStateTool(Tool):
    """
    A pre-built tool that writes a string value to a key in the agent state.

    Only string-typed state keys are supported. This is primarily useful for the agent to surface structured string
    outputs (e.g. a `final_answer` field) that downstream pipeline components can consume.
    """

    def __init__(
        self,
        *,
        name: str = "write_state",
        description: str = "Write a string value to a key in the agent state. Only string-typed keys are supported.",
        key_description: str = (
            "The state key to write. Must be a string-typed key. Call ls_state to see available keys."
        ),
        value_description: str = "The string value to write.",
    ) -> None:
        """
        Initialize the WriteStateTool.

        :param name: Name of the tool.
        :param description: Description of the tool shown to the LLM.
        :param key_description: Description of the `key` parameter shown to the LLM.
        :param value_description: Description of the `value` parameter shown to the LLM.
        """
        self.key_description = key_description
        self.value_description = value_description

        super().__init__(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": key_description},
                    "value": {"type": "string", "description": value_description},
                },
                "required": ["key", "value"],
            },
            function=_write_state,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "name": self.name,
                "description": self.description,
                "key_description": self.key_description,
                "value_description": self.value_description,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteStateTool":
        """
        Deserializes the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        return cls(**data["data"])


StateToolset = Toolset([LsStateTool(), ReadStateTool(), WriteStateTool()])
