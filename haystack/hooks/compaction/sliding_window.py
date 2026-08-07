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


def _messages_from_spans(
    messages: list[ChatMessage], spans: list[tuple[int, int]], skip_compaction_notes: bool = False
) -> list[ChatMessage]:
    """Flatten message spans, optionally excluding messages produced by an earlier compaction."""
    return [
        message
        for start, end in spans
        for message in messages[start:end]
        if not skip_compaction_notes or _COMPACTION_META_KEY not in message.meta
    ]


def _fitting_suffix_start(
    messages: list[ChatMessage],
    spans: list[tuple[int, int]],
    available_tokens: int,
    token_counter: TokenCounter,
    skip_compaction_notes: bool,
) -> int:
    """
    Return the first span in the newest contiguous suffix that fits the token budget.

    Spans are measured from newest to oldest. Retention stops as soon as a span does not fit, ensuring that an older
    span is never kept after a newer one has been removed.

    :param messages: The full conversation containing the messages referenced by `spans`.
    :param spans: Ordered `(start_index, end_index)` pairs to consider for retention. Both indices refer to `messages`,
        and `end_index` is exclusive.
    :param available_tokens: The token budget available for retaining messages from `spans`.
    :param token_counter: The `TokenCounter` used to measure each span.
    :param skip_compaction_notes: Whether messages produced by an earlier compaction are excluded from token counting.
    :returns: The index in `spans` at which the retained suffix begins. If no span fits, returns `len(spans)`; if every
        span fits, returns `0`.
    """
    kept_start = len(spans)
    while kept_start > 0:
        span_messages = _messages_from_spans(
            messages=messages, spans=[spans[kept_start - 1]], skip_compaction_notes=skip_compaction_notes
        )
        span_tokens = token_counter.count(messages=span_messages)
        if span_tokens > available_tokens:
            break
        available_tokens -= span_tokens
        kept_start -= 1
    return kept_start


def _get_turn_start(
    messages: list[ChatMessage],
    available_tokens: int,
    all_current_agent_step_tokens: int,
    historical_turns: list[tuple[int, int]],
    token_counter: TokenCounter,
) -> int:
    """Return the start of the retained historical turns, or `len(historical_turns)` if none fit."""
    kept_turn_start = len(historical_turns)
    if all_current_agent_step_tokens <= available_tokens:
        # Historical turns are considered only when the entire current task fits. This ensures that compaction removes
        # every older turn before it starts trimming individual steps from the task the Agent is actively working on.
        kept_turn_start = _fitting_suffix_start(
            messages=messages,
            spans=historical_turns,
            available_tokens=available_tokens - all_current_agent_step_tokens,
            token_counter=token_counter,
            skip_compaction_notes=True,
        )
    return kept_turn_start


def _get_step_start(
    messages: list[ChatMessage],
    available_tokens: int,
    all_current_agent_step_tokens: int,
    current_agent_steps: list[tuple[int, int]],
    token_counter: TokenCounter,
    min_keep_steps: int,
) -> int:
    """Return the start of the retained current-task steps, or `len(current_agent_steps)` if none fit."""
    if all_current_agent_step_tokens <= available_tokens:
        # Since all current-task steps fit, we can retain all of them.
        return 0
    # Even after dropping every historical turn, the current task is too large. Retain the most recent complete
    # suffix of Agent steps that fits.
    kept_step_start = _fitting_suffix_start(
        messages=messages,
        spans=current_agent_steps,
        available_tokens=available_tokens,
        token_counter=token_counter,
        skip_compaction_notes=False,
    )
    return min(kept_step_start, max(len(current_agent_steps) - min_keep_steps, 0))


def _task_and_step_split(
    messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, min_keep_steps: int
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage], int]:
    """
    Split a conversation into the messages kept before and after an omission note and the messages to remove.

    Leading system messages and the latest real user message are always retained. Historical user turns are retained
    whole when they fit. If the current task itself exceeds the available budget, its oldest Agent steps are removed
    individually while preserving complete assistant/tool-result groups.

    :param messages: The full conversation to split, ordered oldest to newest.
    :param target_tokens: The target token budget for the retained messages.
    :param token_counter: The `TokenCounter` used to decide which historical turns and current-task steps fit.
    :param min_keep_steps: The minimum number of recent current-task Agent steps to retain, even if they exceed the
        target token budget.
    :returns: A tuple containing:

        1. Messages retained before the omission-note position.
        2. Every message selected for removal.
        3. Messages retained after the omission-note position.
        4. The number of retained current-task Agent steps.

        When only historical turns are removed, the first and third elements place the note immediately before the
        current task. When current-task steps are also removed, they place it after the latest user message and before
        the retained current-task steps.
    """
    # Find the leading system messages that contain the Agent instructions.
    system_end = _leading_system_end(messages=messages)

    # Find the latest user message to use as the current task anchor.
    task_index = _latest_user_index(messages=messages)
    task = [messages[task_index]] if task_index is not None else []

    # Find the complete Agent steps that follow the current task anchor.
    current_task_step_start = (task_index + 1) if task_index is not None else system_end
    current_agent_steps = _agent_step_spans(messages=messages, start=current_task_step_start)

    # Find all complete historical turns (i.e. user-assistant) that precede the current task.
    historical_end = task_index if task_index is not None else system_end
    historical_turns = _historical_turn_spans(messages=messages, start=system_end, end=historical_end)

    # Calculate the token count of the protected context (leading system messages and current task)
    protected_tokens = token_counter.count(messages=[*messages[:system_end], *task])

    # Calculate the size of the current task's Agent steps
    all_current_agent_step_tokens = token_counter.count(
        messages=_messages_from_spans(messages=messages, spans=current_agent_steps, skip_compaction_notes=False)
    )

    # Get the start of where we should retain historical turns.
    # If no turns are kept the value of this is `len(historical_turns)`, which is the same as `historical_end`.
    kept_turn_start = _get_turn_start(
        messages=messages,
        available_tokens=target_tokens - protected_tokens,
        all_current_agent_step_tokens=all_current_agent_step_tokens,
        historical_turns=historical_turns,
        token_counter=token_counter,
    )
    kept_turn_spans = historical_turns[kept_turn_start:]
    kept_turns = _messages_from_spans(messages=messages, spans=kept_turn_spans, skip_compaction_notes=True)

    # Get the start of where we should retain current-task steps
    kept_step_start = _get_step_start(
        messages=messages,
        available_tokens=target_tokens - protected_tokens,
        all_current_agent_step_tokens=all_current_agent_step_tokens,
        current_agent_steps=current_agent_steps,
        token_counter=token_counter,
        min_keep_steps=min_keep_steps,
    )
    kept_step_spans = current_agent_steps[kept_step_start:]
    kept_steps = _messages_from_spans(messages=messages, spans=kept_step_spans, skip_compaction_notes=False)

    # Record every index that we are keeping
    kept_indices = {*range(system_end)}
    if task_index is not None:
        kept_indices.add(task_index)
    for start, end in [*kept_turn_spans, *kept_step_spans]:
        kept_indices.update(index for index in range(start, end))
    # Remove everything else that's not in the kept indices
    removable = [message for index, message in enumerate(messages) if index not in kept_indices]

    if kept_step_start == 0:
        kept_before_note = [*messages[:system_end], *kept_turns]
        kept_after_note = [*task, *kept_steps]
    else:
        kept_before_note = [*messages[:system_end], *task]
        kept_after_note = kept_steps
    return kept_before_note, removable, kept_after_note, len(current_agent_steps) - kept_step_start


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
        kept_before_note, removable, kept_after_note, kept_step_count = _task_and_step_split(
            messages=messages,
            target_tokens=target_tokens,
            token_counter=token_counter,
            min_keep_steps=self.min_keep_steps,
        )
        if not removable:
            return None
        if not self.omission_note:
            return [*kept_before_note, *kept_after_note]

        # We prefer user over system since not all providers support multiple system messages
        note = ChatMessage.from_user(
            # `replace` rather than `format`, so a note carrying braces of its own cannot raise.
            self.omission_note.replace(_NUM_REMOVED_PLACEHOLDER, str(len(removable))),
            meta={
                _COMPACTION_META_KEY: {
                    "strategy": "sliding_window",
                    "removed_messages": len(removable),
                    "kept_messages": len(kept_before_note) + len(kept_after_note),
                    "kept_steps": kept_step_count,
                }
            },
        )
        # The note costs tokens of its own, so it is only worth leaving behind if what it stands in for is bigger.
        if token_counter.count([note]) >= token_counter.count(removable):
            return None
        return [*kept_before_note, note, *kept_after_note]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(self, min_keep_steps=self.min_keep_steps, omission_note=self.omission_note)
