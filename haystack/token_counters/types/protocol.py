# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage


class TokenCounter(Protocol):
    """
    Estimates the number tokens used by a list of messages.

    Override `to_dict` when the counter takes constructor arguments, so that they are serialized; the default emits
    none. `from_dict` rebuilds the counter from whatever `to_dict` emitted and rarely needs overriding.
    """

    def count(self, messages: list[ChatMessage]) -> int:
        """
        Return the estimated number of tokens in the given messages.

        :param messages: The messages to measure.
        :returns: The estimated token count.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the counter to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenCounter":
        """Deserialize the counter from a dictionary."""
        return default_from_dict(cls, data)
