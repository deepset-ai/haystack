# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage


class Compactor(Protocol):
    """
    Rewrites an Agent's conversation into a shorter one that carries the same working context.

    A compactor is the *how* of context compaction; deciding *when* to compact is the caller's job, which
    `ContextCompactionHook` does by comparing the context size against a threshold. Strategies differ widely in cost
    and fidelity, from dropping the oldest messages outright to condensing them with an LLM.

    Implementations must honor three rules:

    1. **Return `None` unless the conversation actually gets smaller.** Callers apply whatever else is returned, so
       judging whether compacting was worthwhile is the compactor's job.
    2. **Return a new list; leave `messages` as it is.** The caller owns that list and writes the returned one back.
    3. **Never drop the final message.** The Agent may be mid-step with a pending tool call whose result is appended
       right after compaction returns, and dropping the assistant message holding that call leaves the result orphaned,
       which chat-completion APIs reject.

    Implement both `to_dict` and `from_dict` to make a custom compactor serializable; the default implementations
    below cover compactors whose constructor takes no arguments.
    """

    def compact(self, messages: list[ChatMessage]) -> list[ChatMessage] | None:
        """
        Return a shorter replacement for `messages`, or None to leave it unchanged.

        :param messages: The conversation to compact, oldest to newest.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        ...

    async def compact_async(self, messages: list[ChatMessage]) -> list[ChatMessage] | None:
        """
        Asynchronously return a shorter replacement for `messages`, or None to leave it unchanged.

        The default implementation calls `compact` directly. Override it when compaction does I/O, so the event loop is
        not blocked.

        :param messages: The conversation to compact, oldest to newest.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        return self.compact(messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Compactor":
        """Deserialize the compactor from a dictionary."""
        return default_from_dict(cls, data)
