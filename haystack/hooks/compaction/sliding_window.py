# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _compaction_bounds
from haystack.token_counters import TokenCounter
from haystack.utils.experimental import _experimental

# Placeholder a custom omission note may include to have the number of removed messages substituted in.
_NUM_REMOVED_PLACEHOLDER = "{num_removed}"

_DEFAULT_OMISSION_NOTE = (
    f"[{_NUM_REMOVED_PLACEHOLDER} earlier messages were removed from this conversation to free up context and cannot "
    f"be recovered.]"
)


@_experimental
class SlidingWindowCompactor(Compactor):
    """
    Keeps the leading system messages and as many recent messages as the target allows, dropping everything in between.

    An `omission_note` is left in place of what was removed.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(
        compactor=SlidingWindowCompactor(), context_window=200_000, compact_at=0.7, compact_to=0.4
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
    )
    ```
    """

    def __init__(self, *, min_keep_messages: int = 2, omission_note: str | None = _DEFAULT_OMISSION_NOTE) -> None:
        """
        Initialize the compactor.

        :param min_keep_messages: The fewest recent messages to keep when the target cannot afford more, so a target
            too low still leaves the Agent something to work from. Must be at least 1: the Agent may be mid-step with a
            pending tool call whose result is appended right after compaction.
        :param omission_note: The user message left in place of what was removed, or None to remove the messages
            silently. Include `{num_removed}` to have the number of removed messages substituted in.
        :raises ValueError: If `min_keep_messages` is less than 1.
        """
        if min_keep_messages < 1:
            raise ValueError(
                f"`min_keep_messages` must be at least 1, got {min_keep_messages}. Keeping no messages would drop a "
                f"pending tool call whose result the Agent appends after compaction, leaving it orphaned."
            )
        self.min_keep_messages = min_keep_messages
        self.omission_note = omission_note

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Drop the messages between the leading system block and the retained window.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the retained conversation should come in under.
        :param token_counter: The `TokenCounter` to measure messages with.
        :returns: The system messages, an omission note if one is configured, and the retained window; or None when
            there is nothing worth removing.
        """
        start, end = _compaction_bounds(
            messages, target_tokens=target_tokens, token_counter=token_counter, min_keep_messages=self.min_keep_messages
        )
        removed = end - start
        # With a note, removing a single message just swaps it for the note.
        if removed < (2 if self.omission_note else 1):
            return None

        compacted = list(messages[:start])
        if self.omission_note:
            compacted.append(
                # A user message rather than a system one: providers that hoist system messages into a separate
                # top-level field would move the note away from the point in the conversation it describes.
                ChatMessage.from_user(
                    # `replace` rather than `format`, so a note carrying braces of its own cannot raise.
                    self.omission_note.replace(_NUM_REMOVED_PLACEHOLDER, str(removed)),
                    meta={
                        _COMPACTION_META_KEY: {
                            "strategy": "sliding_window",
                            "removed_messages": removed,
                            "kept_messages": len(messages) - end,
                        }
                    },
                )
            )
        compacted.extend(messages[end:])
        return compacted

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(self, min_keep_messages=self.min_keep_messages, omission_note=self.omission_note)
