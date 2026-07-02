# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.hooks.tool_result_offloading.types import OffloadPolicy
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


class AlwaysOffload(OffloadPolicy):
    """Offload every result of the tool it is assigned to."""

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:  # noqa: ARG002
        """Always returns True."""
        return True


class NeverOffload(OffloadPolicy):
    """Never offload; keep the tool's full result in context. Use to opt a tool out of a wildcard default."""

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:  # noqa: ARG002
        """Always returns False."""
        return False


class OffloadOverChars(OffloadPolicy):
    """Offload a result only when its string length exceeds `threshold` characters."""

    def __init__(self, threshold: int) -> None:
        """
        :param threshold: Offload the result when its length in characters is strictly greater than this value.
        """
        self.threshold = threshold

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:  # noqa: ARG002
        """Return whether `result` is longer than `threshold` characters."""
        return len(result) > self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy, including its threshold."""
        return default_to_dict(self, threshold=self.threshold)


class CallableOffloadPolicy(OffloadPolicy):
    """Offload based on a user-supplied `(tool_name, result, state) -> bool` condition."""

    def __init__(self, condition: Callable[[str, str, State], bool]) -> None:
        """
        Initialize the policy with its condition callable.

        :param condition: Callable receiving the tool name, the result string, and the live `State`; return True to
            offload. It must be serializable (a module-level function or an importable callable).
        """
        self.condition = condition

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:
        """Delegate the decision to the wrapped condition."""
        return self.condition(tool_name, result, state)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy, encoding its condition callable."""
        return default_to_dict(self, condition=serialize_callable(self.condition))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallableOffloadPolicy":
        """Deserialize the policy, restoring its condition callable."""
        data["init_parameters"]["condition"] = deserialize_callable(data["init_parameters"]["condition"])
        return default_from_dict(cls, data)
