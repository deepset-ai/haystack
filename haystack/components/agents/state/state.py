# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from haystack.dataclasses import ChatMessage
from haystack.utils import _deserialize_value_with_schema, _serialize_value_with_schema
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.type_serialization import deserialize_type, serialize_type

from .state_utils import _is_list_type, _is_valid_type, merge_lists, replace_values


def _schema_to_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a schema dictionary to a serializable format.

    Converts each parameter's type and optional handler function into a serializable
    format using type and callable serialization utilities.

    :param schema: Dictionary mapping parameter names to their type and handler configs
    :returns: Dictionary with serialized type and handler information
    """
    serialized_schema = {}
    for param, config in schema.items():
        serialized_schema[param] = {"type": serialize_type(config["type"])}
        if config.get("handler"):
            serialized_schema[param]["handler"] = serialize_callable(config["handler"])

    return serialized_schema


def _schema_from_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a serialized schema dictionary back to its original format.

    Deserializes the type and optional handler function for each parameter from their
    serialized format back into Python types and callables.

    :param schema: Dictionary containing serialized schema information
    :returns: Dictionary with deserialized type and handler configurations
    """
    deserialized_schema = {}
    for param, config in schema.items():
        deserialized_schema[param] = {"type": deserialize_type(config["type"])}

        if config.get("handler"):
            deserialized_schema[param]["handler"] = deserialize_callable(config["handler"])

    return deserialized_schema


def _validate_schema(schema: Dict[str, Any]) -> None:
    """
    Validate that a schema dictionary meets all required constraints.

    Checks that each parameter definition has a valid type field and that any handler
    specified is a callable function.

    :param schema: Dictionary mapping parameter names to their type and handler configs
    :raises ValueError: If schema validation fails due to missing or invalid fields
    """
    for param, definition in schema.items():
        if "type" not in definition:
            raise ValueError(f"StateSchema: Key '{param}' is missing a 'type' entry.")
        if not _is_valid_type(definition["type"]):
            raise ValueError(f"StateSchema: 'type' for key '{param}' must be a Python type, got {definition['type']}")
        if definition.get("handler") is not None and not callable(definition["handler"]):
            raise ValueError(f"StateSchema: 'handler' for key '{param}' must be callable or None")
        if param == "messages" and definition["type"] != List[ChatMessage]:
            raise ValueError(f"StateSchema: 'messages' must be of type List[ChatMessage], got {definition['type']}")


class State:
    """
    A class that wraps a StateSchema and maintains an internal _data dictionary.

    Each schema entry has:
      "parameter_name": {
        "type": SomeType,
        "handler": Optional[Callable[[Any, Any], Any]]
      }
    """

    def __init__(self, schema: Dict[str, Any], data: Optional[Dict[str, Any]] = None):
        """
        Initialize a State object with a schema and optional data.

        :param schema: Dictionary mapping parameter names to their type and handler configs.
            Type must be a valid Python type, and handler must be a callable function or None.
            If handler is None, the default handler for the type will be used. The default handlers are:
                - For list types: `haystack.agents.state.state_utils.merge_lists`
                - For all other types: `haystack.agents.state.state_utils.replace_values`
        :param data: Optional dictionary of initial data to populate the state
        """
        _validate_schema(schema)
        self.schema = deepcopy(schema)
        if self.schema.get("messages") is None:
            self.schema["messages"] = {"type": List[ChatMessage], "handler": merge_lists}
        self._data = data or {}

        # Set default handlers if not provided in schema
        for definition in self.schema.values():
            # Skip if handler is already defined and not None
            if definition.get("handler") is not None:
                continue
            # Set default handler based on type
            if _is_list_type(definition["type"]):
                definition["handler"] = merge_lists
            else:
                definition["handler"] = replace_values

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the state by key.

        :param key: Key to look up in the state
        :param default: Value to return if key is not found
        :returns: Value associated with key or default if not found
        """
        return deepcopy(self._data.get(key, default))

    def set(self, key: str, value: Any, handler_override: Optional[Callable[[Any, Any], Any]] = None) -> None:
        """
        Set or merge a value in the state according to schema rules.

        Value is merged or overwritten according to these rules:
          - if handler_override is given, use that
          - else use the handler defined in the schema for 'key'

        :param key: Key to store the value under
        :param value: Value to store or merge
        :param handler_override: Optional function to override the default merge behavior
        """
        # If key not in schema, we throw an error
        definition = self.schema.get(key, None)
        if definition is None:
            raise ValueError(f"State: Key '{key}' not found in schema. Schema: {self.schema}")

        # Get current value from state and apply handler
        current_value = self._data.get(key, None)
        handler = handler_override or definition["handler"]
        self._data[key] = handler(current_value, value)

    @property
    def data(self):
        """
        All current data of the state.
        """
        return self._data

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the state.

        :param key: Key to check for existence
        :returns: True if key exists in state, False otherwise
        """
        return key in self._data

    def to_dict(self):
        """
        Convert the State object to a dictionary.
        """
        serialized = {}
        serialized["schema"] = _schema_to_dict(self.schema)
        serialized["data"] = _serialize_value_with_schema(self._data)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Convert a dictionary back to a State object.
        """
        schema = _schema_from_dict(data.get("schema", {}))
        deserialized_data = _deserialize_value_with_schema(data.get("data", {}))
        return State(schema, deserialized_data)
