# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict
from haystack.dataclasses import ChatMessage
from haystack.tools import ToolsType


class TokenCounter(Protocol):
    """
    Estimates the number tokens used by a list of messages.

    Implement `to_dict` so the counter's settings survive serialization. The default `from_dict` passes them straight
    back to the constructor, which is enough for plain values; override it when `to_dict` emitted something that has to
    be rebuilt first, such as a `Secret` or a nested component.
    """

    def count(self, messages: list[ChatMessage], tools: ToolsType | None = None) -> int:
        """
        Return the estimated number of tokens in the given messages.

        :param messages: The messages to measure.
        :param tools: Tools whose schemas are sent alongside the messages, and so consume tokens too. Pass them to have
            them counted; leave as None to measure the messages alone.
        :returns: The estimated token count.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the counter to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenCounter":
        """Deserialize the counter from a dictionary."""
        return default_from_dict(cls, data)
