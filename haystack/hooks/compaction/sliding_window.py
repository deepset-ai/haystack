# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _preserved_prefix_end, _safe_cut_index


class SlidingWindowCompactor(Compactor):
    """
    Keeps the leading system messages and the most recent messages, dropping everything in between.

    The cheapest strategy and the most lossy: nothing is written in place of what it removes, so anything the Agent
    learned outside the retained window is simply gone. It suits runs whose turns are largely independent, or as a
    last resort behind a strategy that preserves more.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(compactor=SlidingWindowCompactor(keep_last_n_messages=20), threshold_tokens=100_000)
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
    )
    ```
    """

    def __init__(self, *, keep_last_n_messages: int = 20, omission_note: bool = True) -> None:
        """
        Initialize the compactor with the size of the window it retains.

        :param keep_last_n_messages: How many of the most recent messages to keep. The boundary is moved further back
            when it would otherwise split a tool call from its results, so slightly more may be kept. Must be at least
            1: the Agent may be mid-step with a pending tool call whose result is appended right after compaction, and
            dropping the message holding that call would leave the result orphaned.
        :param omission_note: Whether to leave a short system message in place of what was removed. Keep it on unless
            you have a reason not to - without it the conversation reads as though the removed turns never happened,
            and the model may repeat work or answer as if it had context it no longer has.
        :raises ValueError: If `keep_last_n_messages` is less than 1.
        """
        if keep_last_n_messages < 1:
            raise ValueError(
                f"`keep_last_n_messages` must be at least 1, got {keep_last_n_messages}. Keeping no messages would "
                f"drop a pending tool call whose result the Agent appends after compaction, leaving it orphaned."
            )
        self.keep_last_n_messages = keep_last_n_messages
        self.omission_note = omission_note

    def compact(self, state: State) -> list[ChatMessage] | None:
        """
        Drop the messages between the leading system block and the retained window.

        :param state: The Agent's `State`, read only. The conversation is `state.data["messages"]`.
        :returns: The leading system messages, an optional omission note, and the retained window; or None when the
            conversation already fits in the window, or when the omission note would cost as much as it saves.
        """
        messages = state.data.get("messages") or []
        prefix_end = _preserved_prefix_end(messages)
        cut = _safe_cut_index(messages, prefix_end, self.keep_last_n_messages)
        if cut <= prefix_end:
            return None
        # Dropping a single message only to add a note in its place gains nothing, and would then repeat every step,
        # replacing one note with the next.
        if self.omission_note and cut - prefix_end < 2:
            return None

        compacted = list(messages[:prefix_end])
        if self.omission_note:
            removed = cut - prefix_end
            subject = "1 earlier message was" if removed == 1 else f"{removed} earlier messages were"
            compacted.append(
                ChatMessage.from_system(
                    f"[{subject} removed from this conversation to free up context and cannot be recovered.]",
                    meta={
                        _COMPACTION_META_KEY: {
                            "step": state.data.get("step_count", 0),
                            "strategy": "sliding_window",
                            "removed_messages": removed,
                            "kept_messages": len(messages) - cut,
                        }
                    },
                )
            )
        compacted.extend(messages[cut:])
        return compacted

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(self, keep_last_n_messages=self.keep_last_n_messages, omission_note=self.omission_note)
