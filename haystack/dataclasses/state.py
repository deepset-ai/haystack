# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import warnings
from typing import Any, Dict, Optional

from haystack.components.agents import State as Utils_State


class State(Utils_State):
    """
    A class that wraps a StateSchema and maintains an internal _data dictionary.

    Deprecated in favor of `haystack.components.agents.State`. It will be removed in Haystack 2.16.0.

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
                - For list types: `haystack.components.agents.state.state_utils.merge_lists`
                - For all other types: `haystack.components.agents.state.state_utils.replace_values`
        :param data: Optional dictionary of initial data to populate the state
        """

        warnings.warn(
            "`haystack.dataclasses.State` is deprecated and will be removed in Haystack 2.16.0. "
            "Use `haystack.components.agents.State` instead.",
            DeprecationWarning,
        )

        super().__init__(schema, data)
