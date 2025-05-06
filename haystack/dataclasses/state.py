# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import warnings
from typing import Any, Callable, Dict, Optional

from haystack.utils.state import State as Utils_State


class State(Utils_State):
    """
    A class that wraps a StateSchema and maintains an internal _data dictionary.

    Deprecated in favor of `haystack.utils.State`. It will be removed in Haystack 2.16.0.

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
                - For list types: `haystack.dataclasses.state_utils.merge_lists`
                - For all other types: `haystack.dataclasses.state_utils.replace_values`
        :param data: Optional dictionary of initial data to populate the state
        """

        warnings.warn(
            "`haystack.dataclasses.State` is deprecated and will be removed in Haystack 2.16.0. "
            "Use `haystack.utils.State` instead.",
            DeprecationWarning,
        )

        super().__init__(schema, data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the state by key.

        :param key: Key to look up in the state
        :param default: Value to return if key is not found
        :returns: Value associated with key or default if not found
        """
        return super().get(key, default)

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
        super().set(key, value, handler_override)

    @property
    def data(self):
        """
        All current data of the state.
        """
        return super().data

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the state.

        :param key: Key to check for existence
        :returns: True if key exists in state, False otherwise
        """
        return super().has(key)

    def to_dict(self):
        """
        Convert the State object to a dictionary.
        """
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Convert a dictionary back to a State object.
        """
        return super().from_dict(data)
