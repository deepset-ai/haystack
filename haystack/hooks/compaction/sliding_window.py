# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _compaction_bounds

# Placeholder a custom omission note may include to have the number of removed messages substituted in.
_NUM_REMOVED_PLACEHOLDER = "{num_removed}"

_DEFAULT_OMISSION_NOTE = (
    f"[{_NUM_REMOVED_PLACEHOLDER} earlier messages were removed from this conversation to free up context and cannot "
    f"be recovered.]"
)


class SlidingWindowCompactor(Compactor):
    """
    Keeps the leading system messages and the most recent messages, dropping everything in between.

    An `omission_note` is left in place of what was removed. Cheap but lossy: whatever the Agent learned outside the
    retained window is gone, so it suits runs whose turns are largely independent.

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

    def __init__(self, *, keep_last_n_messages: int = 20, omission_note: str | None = _DEFAULT_OMISSION_NOTE) -> None:
        """
        Initialize the compactor with the size of the window it retains.

        :param keep_last_n_messages: How many of the most recent messages to keep. Slightly more may be kept when the
            boundary would otherwise split a tool call from its results.
        :param omission_note: The user message left in place of what was removed, or None to remove the messages
            silently. Include `{num_removed}` to have the number of removed messages substituted in.
        :raises ValueError: If `keep_last_n_messages` is less than 1, which would drop a pending tool call whose result
            the Agent appends right after compaction.
        """
        if keep_last_n_messages < 1:
            raise ValueError(
                f"`keep_last_n_messages` must be at least 1, got {keep_last_n_messages}. Keeping no messages would "
                f"drop a pending tool call whose result the Agent appends after compaction, leaving it orphaned."
            )
        self.keep_last_n_messages = keep_last_n_messages
        self.omission_note = omission_note

    def compact(self, messages: list[ChatMessage]) -> list[ChatMessage] | None:
        """
        Drop the messages between the leading system block and the retained window.

        :param messages: The conversation to compact, oldest to newest.
        :returns: The system messages, an omission note if one is configured, and the retained window; or None when
            there is nothing worth removing.
        """
        start, end = _compaction_bounds(messages, self.keep_last_n_messages)
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
        return default_to_dict(self, keep_last_n_messages=self.keep_last_n_messages, omission_note=self.omission_note)
