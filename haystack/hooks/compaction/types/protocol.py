# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict
from haystack.dataclasses import ChatMessage
from haystack.token_counters import TokenCounter


class Compactor(Protocol):
    """
    Rewrites an Agent's conversation into a shorter one that carries the same working context.

    A compactor is the *how* of context compaction; deciding *when* to compact is the caller's job, which
    `CompactionHook` does by comparing the context size against a fraction of the model's window. Strategies
    differ widely in cost and fidelity, from dropping the oldest messages outright to condensing them with an LLM.

    Implementations must honor three rules:

    1. **Return `None` unless the conversation actually gets smaller.** Callers apply whatever else is returned, so
       judging whether compacting was worthwhile is the compactor's job.
    2. **Return a new list; leave `messages` as it is.** The caller owns that list and writes the returned one back.
    3. **Keep tool calls and their results together.** Do not retain a tool result after removing the assistant message
       that contains its originating call, or retain a tool call without all of its results. Chat-completion APIs reject
       these incomplete tool-call exchanges.

    `target_tokens` is a goal, not a guarantee: a compactor that cannot reach it should get as close as it can rather
    than strip the conversation past what the Agent needs to keep working.

    Implement `to_dict` so the compactor's settings survive serialization. The default `from_dict` passes them straight
    back to the constructor, which is enough for plain values; override it when `to_dict` emitted something that has to
    be rebuilt first, such as a `Secret` or a nested component.
    """

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Return a shorter replacement for `messages`, or None to leave it unchanged.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` to measure messages with. The same one the caller sized the context
            with, so a compactor's measurements are consistent with the decision to compact.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        ...

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Asynchronously return a shorter replacement for `messages`, or None to leave it unchanged.

        The default implementation calls `compact` directly. Override it when compaction does I/O, so the event loop is
        not blocked.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` to measure messages with. The same one the caller sized the context
            with, so a compactor's measurements are consistent with the decision to compact.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        return self.compact(messages, target_tokens, token_counter)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Compactor":
        """Deserialize the compactor from a dictionary."""
        return default_from_dict(cls, data)
