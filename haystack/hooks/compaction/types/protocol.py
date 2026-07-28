# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage


class Compactor(Protocol):
    """
    Rewrites an Agent's conversation into a shorter one that carries the same working context.

    A compactor is the *how* of context compaction; deciding *when* to compact is the caller's job (the
    `ContextCompactionHook` compares the context size against a threshold, and `ContextCompactionTool` lets the model
    ask for it). Strategies differ widely in cost and fidelity: summarizing the older turns with an LLM, replacing
    stale tool results with a placeholder, or dropping the oldest messages outright.

    Implementations must honor three rules:

    1. **Return `None` when there is nothing to do.** Callers leave the conversation untouched in that case, so a
       compactor that cannot shrink the conversation any further costs nothing to call repeatedly.
    2. **Do not mutate `state`.** It is read-only context: `messages` to compact, plus `step_count`,
       `context_tokens`, `tools`, and `hook_context` for strategies that want them. The caller writes the returned
       list back into `State`.
    3. **Never drop the final message.** The Agent may be mid-step with a pending tool call whose result is appended
       right after compaction returns; dropping the assistant message that holds that call leaves the result orphaned,
       which chat-completion APIs reject. Keeping a non-empty tail of recent messages, or rewriting messages in place,
       satisfies this.

    Implement both `to_dict` and `from_dict` to make a custom compactor serializable; the default implementations
    below cover compactors whose constructor takes no arguments.
    """

    def compact(self, state: State) -> list[ChatMessage] | None:
        """
        Return a compacted replacement for `state.data["messages"]`, or None to leave it unchanged.

        :param state: The Agent's `State`, read-only. The conversation to compact is `state.data["messages"]`.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        ...

    async def compact_async(self, state: State) -> list[ChatMessage] | None:
        """
        Asynchronously return a compacted replacement for `state.data["messages"]`, or None to leave it unchanged.

        The default implementation calls `compact` directly, which is correct for a strategy that only rearranges
        messages. Override it when compaction does I/O — for example, calling a Chat Generator to write a summary —
        so the event loop is not blocked.

        :param state: The Agent's `State`, read-only. The conversation to compact is `state.data["messages"]`.
        :returns: The replacement conversation, or None when this compactor has nothing to change.
        """
        return self.compact(state)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Compactor":
        """Deserialize the compactor from a dictionary."""
        return default_from_dict(cls, data)
