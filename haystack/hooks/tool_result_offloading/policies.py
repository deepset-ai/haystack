# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_to_dict
from haystack.hooks.tool_result_offloading.types import OffloadPolicy


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
