# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Protocol

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.token_counters import TokenCounter


@dataclass(frozen=True)
class CompactionBudget:
    """
    How small a compactor should make the conversation, and the counter to measure it with.

    :param target_tokens: The size the compacted conversation should come in under.
    :param counter: The `TokenCounter` to measure messages with. The same one the hook used to decide to compact, so a
        compactor's measurements are consistent with the trigger's.
    """

    target_tokens: int
    counter: TokenCounter


class Compactor(Protocol):
    """
    Rewrites an Agent's conversation into a shorter one that carries the same working context.

    A compactor is the *how* of context compaction; deciding *when* to compact is the caller's job, which
    `ContextCompactionHook` does by comparing the context size against a fraction of the model's window. Strategies
    differ widely in cost and fidelity, from dropping the oldest messages outright to condensing them with an LLM.

    Implementations must honor three rules:

    1. **Return `None` unless the conversation actually gets smaller.** Callers apply whatever else is returned, so
       judging whether compacting was worthwhile is the compactor's job.
    2. **Return a new list; leave `messages` as it is.** The caller owns that list and writes the returned one back.
    3. **Never drop the final message.** The Agent may be mid-step with a pending tool call whose result is appended
       right after compaction returns, and dropping the assistant message holding that call leaves the result orphaned,
       which chat-completion APIs reject.

    `budget.target_tokens` is a goal, not a guarantee: a compactor that cannot reach it should get as close as it can
    rather than strip the conversation past what the Agent needs to keep working.

    Implement both `to_dict` and `from_dict` to make a custom compactor serializable; the default implementations
    below cover compactors whose constructor takes no arguments.
    """

    def compact(self, messages: list[ChatMessage], budget: CompactionBudget) -> list[ChatMessage] | None:
        """
        Return a shorter replacement for `messages`, or None to leave it unchanged.

        :param messages: The conversation to compact, oldest to newest.
        :param budget: The size to aim for and the counter to measure with.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        ...

    async def compact_async(self, messages: list[ChatMessage], budget: CompactionBudget) -> list[ChatMessage] | None:
        """
        Asynchronously return a shorter replacement for `messages`, or None to leave it unchanged.

        The default implementation calls `compact` directly. Override it when compaction does I/O, so the event loop is
        not blocked.

        :param messages: The conversation to compact, oldest to newest.
        :param budget: The size to aim for and the counter to measure with.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        return self.compact(messages, budget)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Compactor":
        """Deserialize the compactor from a dictionary."""
        return default_from_dict(cls, data)
