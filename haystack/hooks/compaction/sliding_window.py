# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _agent_step_spans
from haystack.token_counters import TokenCounter
from haystack.utils.experimental import _experimental

# Placeholder a custom omission note may include to have the number of removed messages substituted in.
_NUM_REMOVED_PLACEHOLDER = "{num_removed}"

_DEFAULT_OMISSION_NOTE = (
    f"[{_NUM_REMOVED_PLACEHOLDER} earlier messages were removed from this conversation to free up context and cannot "
    f"be recovered.]"
)


def _leading_system_end(messages: list[ChatMessage]) -> int:
    """Return the end of the leading system-message block."""
    for index, message in enumerate(messages):
        # Find the first non-leading system message or a system message produced by compaction
        if not message.is_from(role=ChatRole.SYSTEM) or _COMPACTION_META_KEY in message.meta:
            return index
    return len(messages)


def _latest_user_index(messages: list[ChatMessage]) -> int | None:
    """
    Return the latest user message not produced by compaction.

    :param messages: The conversation to analyze, oldest to newest.
    """
    # We loop backwards to find the latest user message
    for index in reversed(range(len(messages))):
        message = messages[index]
        # Find the latest user message that was not produced by a previous compaction
        if message.is_from(role=ChatRole.USER) and _COMPACTION_META_KEY not in message.meta:
            return index
    return None


def _task_and_step_split(
    messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, min_keep_steps: int
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage], int]:
    """Split messages into the protected task context, removable history, and retained Agent steps."""
    # Find the leading system messages that contain the Agent instructions.
    system_end = _leading_system_end(messages=messages)
    # Find the latest user message to use as the current task anchor.
    task_index = _latest_user_index(messages=messages)
    task = [messages[task_index]] if task_index is not None else []
    # Find complete Agent steps after the current task, or after the system messages when there is no task anchor.
    steps = _agent_step_spans(messages=messages, start=(task_index + 1) if task_index is not None else system_end)

    # Protect the Agent instructions and current task from removal.
    protected = [*messages[:system_end], *task]
    # The remaining token budget to retain recent Agent steps.
    available_tokens = target_tokens - token_counter.count(messages=protected)
    # Work backwards through the steps, keeping as many as fit in the remaining budget.
    kept_step_start = len(steps)
    while kept_step_start > 0:
        start, end = steps[kept_step_start - 1]
        step_tokens = token_counter.count(messages=messages[start:end])
        # Stop at the first step that does not fit
        if step_tokens > available_tokens:
            break
        available_tokens -= step_tokens
        kept_step_start -= 1

    # Enforce the minimum number of complete steps, even when they exceed the target token budget.
    kept_step_start = min(kept_step_start, max(len(steps) - min_keep_steps, 0))
    kept_spans = steps[kept_step_start:]

    # Record every protected or retained message index; equal ChatMessages can appear more than once in the list.
    kept_indices = {*range(system_end)}
    if task_index is not None:
        kept_indices.add(task_index)
    for start, end in kept_spans:
        kept_indices.update(range(start, end))

    # Everything outside the protected context and retained steps can be removed.
    removable = [message for index, message in enumerate(messages) if index not in kept_indices]
    # Flatten the retained spans back into the message list expected by the compactor.
    kept_steps = [message for start, end in kept_spans for message in messages[start:end]]
    return protected, removable, kept_steps, len(kept_spans)


@_experimental
class SlidingWindowCompactor(Compactor):
    """
    Keeps the Agent's instructions, current task, and as many complete recent steps as the target allows.

    Leading system messages and the latest user message are protected. Recent history is retained in complete Agent
    steps, where a step is an assistant message together with all immediately following tool results. An
    `omission_note` is left in place of what was removed.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(
        compactor=SlidingWindowCompactor(), context_window=400_000, compact_at=0.7, compact_to=0.4
    )
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
    )
    ```
    """

    def __init__(self, *, min_keep_steps: int = 1, omission_note: str | None = _DEFAULT_OMISSION_NOTE) -> None:
        """
        Initialize the compactor.

        :param min_keep_steps: The fewest complete recent Agent steps to keep even when they exceed the target. A step
            is an assistant message and all immediately following tool results. `0` allows all completed steps to be
            removed when none fit.
        :param omission_note: The user message left in place of what was removed, or None to remove the messages
            silently. Include `{num_removed}` to have the number of removed messages substituted in.
        :raises ValueError: If `min_keep_steps` is negative.
        """
        if min_keep_steps < 0:
            raise ValueError(f"`min_keep_steps` must be at least 0, got {min_keep_steps}.")
        self.min_keep_steps = min_keep_steps
        self.omission_note = omission_note

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Drop older history while preserving the task anchor and a recent window of complete Agent steps.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the retained conversation should come in under.
        :param token_counter: The `TokenCounter` to measure messages with.
        :returns: The protected context, an omission note if configured, and the retained steps; or None when there is
            nothing worth removing.
        """
        if token_counter.count(messages) <= target_tokens:
            return None
        protected, removable, kept_steps, kept_step_count = _task_and_step_split(
            messages=messages,
            target_tokens=target_tokens,
            token_counter=token_counter,
            min_keep_steps=self.min_keep_steps,
        )
        if not removable:
            return None
        if not self.omission_note:
            return [*protected, *kept_steps]

        # We prefer user over system since not all providers support multiple system messages
        note = ChatMessage.from_user(
            # `replace` rather than `format`, so a note carrying braces of its own cannot raise.
            self.omission_note.replace(_NUM_REMOVED_PLACEHOLDER, str(len(removable))),
            meta={
                _COMPACTION_META_KEY: {
                    "strategy": "sliding_window",
                    "removed_messages": len(removable),
                    "kept_messages": len(protected) + len(kept_steps),
                    "kept_steps": kept_step_count,
                }
            },
        )
        # The note costs tokens of its own, so it is only worth leaving behind if what it stands in for is bigger.
        if token_counter.count([note]) >= token_counter.count(removable):
            return None
        return [*protected, note, *kept_steps]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(self, min_keep_steps=self.min_keep_steps, omission_note=self.omission_note)
