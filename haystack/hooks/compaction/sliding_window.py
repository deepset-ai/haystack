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


def _historical_turn_spans(messages: list[ChatMessage], start: int, end: int) -> list[tuple[int, int]]:
    """
    Return spans for complete user turns in a bounded section of conversation history.

    Each turn begins with a real user message and continues up to, but does not include, the next real user message.
    This groups a user's request with every assistant step and tool result produced in response to it.

    :param messages: The full conversation to analyze, ordered oldest to newest.
    :param start: The inclusive index at which to begin looking for historical turns.
    :param end: The exclusive index at which to stop. This is normally the current task's user-message index.
    :returns: Ordered `(start_index, end_index)` pairs for each complete historical turn. Both indices refer to
        `messages`, and `end_index` is exclusive, so a returned pair can be used directly as `messages[start:end]`.
    """
    # Compaction notes use the user role for provider compatibility, but they do not begin a new conversation turn.
    # Ignoring marked messages here also lets a subsequent compaction fold an old note into its replacement.
    user_indices = [
        index
        for index in range(start, end)
        if messages[index].is_from(role=ChatRole.USER) and _COMPACTION_META_KEY not in messages[index].meta
    ]

    # A real user message closes the preceding turn and starts the next one. The final historical turn extends to the
    # supplied boundary, which is typically where the protected current task begins.
    return [
        (index, user_indices[position + 1] if position + 1 < len(user_indices) else end)
        for position, index in enumerate(user_indices)
    ]


def _task_and_step_split(
    messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, min_keep_steps: int
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage], int]:
    """Split messages into the protected prefix, removable history, and retained conversation window."""
    # Find the leading system messages that contain the Agent instructions.
    system_end = _leading_system_end(messages=messages)
    # Find the latest user message to use as the current task anchor.
    task_index = _latest_user_index(messages=messages)
    task = [messages[task_index]] if task_index is not None else []
    step_start = (task_index + 1) if task_index is not None else system_end
    # Current-task steps can be removed individually. Earlier user/assistant exchanges are kept as complete turns so
    # an assistant reply is never retained without the user message it answers.
    steps = _agent_step_spans(messages=messages, start=step_start)
    historical_end = task_index if task_index is not None else system_end
    historical_turns = _historical_turn_spans(messages, system_end, historical_end)

    # Protect the Agent instructions and current task from removal.
    protected = [*messages[:system_end], *task]
    # The remaining token budget after protecting the instructions and current task.
    available_tokens = target_tokens - token_counter.count(messages=protected)
    kept_turn_start = len(historical_turns)
    all_step_tokens = token_counter.count(messages=[message for start, end in steps for message in messages[start:end]])
    if all_step_tokens <= available_tokens:
        # Historical turns are considered only when the entire current task fits. This ensures that compaction removes
        # every older turn before it starts trimming individual steps from the task the Agent is actively working on.
        kept_step_start = 0
        available_tokens -= all_step_tokens
        while kept_turn_start > 0:
            start, end = historical_turns[kept_turn_start - 1]
            turn = [message for message in messages[start:end] if _COMPACTION_META_KEY not in message.meta]
            turn_tokens = token_counter.count(messages=turn)
            if turn_tokens > available_tokens:
                break
            available_tokens -= turn_tokens
            kept_turn_start -= 1
    else:
        # Even after dropping every historical turn, the current task is too large. Work backwards through its Agent
        # steps and retain the most recent complete suffix that fits.
        kept_step_start = len(steps)
        while kept_step_start > 0:
            start, end = steps[kept_step_start - 1]
            step_tokens = token_counter.count(messages=messages[start:end])
            if step_tokens > available_tokens:
                break
            available_tokens -= step_tokens
            kept_step_start -= 1

    # Enforce the minimum number of complete steps, even when they exceed the target token budget.
    kept_step_start = min(kept_step_start, max(len(steps) - min_keep_steps, 0))
    kept_step_spans = steps[kept_step_start:]
    kept_turn_spans = historical_turns[kept_turn_start:]

    # Record every protected or retained message index; equal ChatMessages can appear more than once in the list.
    kept_indices = {*range(system_end)}
    if task_index is not None:
        kept_indices.add(task_index)
    for start, end in [*kept_turn_spans, *kept_step_spans]:
        kept_indices.update(index for index in range(start, end) if _COMPACTION_META_KEY not in messages[index].meta)

    # Everything outside the protected context and retained steps can be removed.
    removable = [message for index, message in enumerate(messages) if index not in kept_indices]
    kept_turns = [
        message
        for start, end in kept_turn_spans
        for message in messages[start:end]
        if _COMPACTION_META_KEY not in message.meta
    ]
    kept_steps = [message for start, end in kept_step_spans for message in messages[start:end]]
    return [*messages[:system_end], *kept_turns, *task], removable, kept_steps, len(steps) - kept_step_start


@_experimental
class SlidingWindowCompactor(Compactor):
    """
    Keeps the Agent's instructions, current task, and as much complete recent conversation as the target allows.

    Leading system messages and the latest user message are protected. Earlier user/assistant turns are retained when
    they fit, and the current task's history is retained in complete Agent steps, where a step is an assistant message
    together with all immediately following tool results. An `omission_note` is left in place of what was removed.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, SlidingWindowCompactor

    hook = CompactionHook(
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
        Drop older history while preserving the task anchor and a complete recent conversation window.

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
