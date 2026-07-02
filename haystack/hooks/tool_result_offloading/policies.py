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
        """
        Decide whether to offload the given tool result.

        :param tool_name: The name of the tool that produced the result (unused; this policy always offloads).
        :param result: The tool result string (unused; this policy always offloads).
        :param state: The Agent's live `State` (unused; this policy always offloads).
        :returns: Always True.
        """
        return True


class NeverOffload(OffloadPolicy):
    """Never offload; keep the tool's full result in context. Use to opt a tool out of a wildcard default."""

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:  # noqa: ARG002
        """
        Decide whether to offload the given tool result.

        :param tool_name: The name of the tool that produced the result (unused; this policy never offloads).
        :param result: The tool result string (unused; this policy never offloads).
        :param state: The Agent's live `State` (unused; this policy never offloads).
        :returns: Always False.
        """
        return False


class OffloadOverChars(OffloadPolicy):
    """Offload a result only when its string length exceeds `threshold` characters."""

    def __init__(self, threshold: int) -> None:
        """
        Initialize the policy with its character threshold.

        :param threshold: Offload the result when its length in characters is strictly greater than this value.
        """
        self.threshold = threshold

    def should_offload(self, tool_name: str, result: str, state: State) -> bool:  # noqa: ARG002
        """
        Decide whether to offload the given tool result based on its length.

        :param tool_name: The name of the tool that produced the result (unused; only length is considered).
        :param result: The tool result string whose length is compared against the threshold.
        :param state: The Agent's live `State` (unused; only length is considered).
        :returns: True when `result` is longer than `threshold` characters, otherwise False.
        """
        return len(result) > self.threshold

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the policy, including its threshold.

        :returns: A dictionary representation of the policy.
        """
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
        """
        Delegate the offload decision to the wrapped condition.

        :param tool_name: The name of the tool that produced the result.
        :param result: The tool result string.
        :param state: The Agent's live `State`.
        :returns: Whatever the wrapped condition returns for these arguments.
        """
        return self.condition(tool_name, result, state)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the policy, encoding its condition callable.

        :returns: A dictionary representation of the policy.
        """
        return default_to_dict(self, condition=serialize_callable(self.condition))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallableOffloadPolicy":
        """
        Deserialize the policy, restoring its condition callable.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `CallableOffloadPolicy`.
        """
        data["init_parameters"]["condition"] = deserialize_callable(data["init_parameters"]["condition"])
        return default_from_dict(cls, data)
