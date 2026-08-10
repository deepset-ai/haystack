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

# Recorded as the strategy on every message this compactor produces, so a later run can recognize its own notes.
_STRATEGY = "sliding_window"

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


def _is_compaction_note(message: ChatMessage) -> bool:
    """Whether a message is an omission note this strategy left in place of removed history."""
    marker = message.meta.get(_COMPACTION_META_KEY)
    return message.is_from(role=ChatRole.USER) and isinstance(marker, dict) and marker.get("strategy") == _STRATEGY


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
    # Reject any user-role message an earlier compaction produced, whichever strategy made it: none of them are user
    # requests, so none of them begin a turn. Leaving them out also lets this compaction fold an old note away.
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


def _index_groups(
    messages: list[ChatMessage], spans: list[tuple[int, int]], skip_compaction_notes: bool = False
) -> list[list[int]]:
    """
    Expand each span into the message indices it covers, optionally dropping messages an earlier compaction produced.
    """
    return [
        [index for index in range(start, end) if not (skip_compaction_notes and _is_compaction_note(messages[index]))]
        for start, end in spans
    ]


def _messages_at(messages: list[ChatMessage], indices: list[int]) -> list[ChatMessage]:
    """Return the messages at the given indices, in the order the indices are given."""
    return [messages[index] for index in indices]


def _messages_except(messages: list[ChatMessage], indices: list[int]) -> list[ChatMessage]:
    """Return the messages the given indices leave out, in conversation order."""
    left_out = set(indices)
    return [message for index, message in enumerate(messages) if index not in left_out]


def _flatten(groups: list[list[int]]) -> list[int]:
    """Join index groups into a single ordered list of indices."""
    return [index for group in groups for index in group]


def _removable_groups(
    messages: list[ChatMessage], system_end: int, task_index: int | None
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Group the two stretches of conversation compaction is allowed to remove.

    :param messages: The full conversation, ordered oldest to newest.
    :param system_end: The end of the leading system-message block.
    :param task_index: The index of the user message anchoring the current task, or None when there is none.
    :returns: Index groups for the complete historical turns preceding the current task, and index groups for the
        current task's own Agent steps. Both are ordered oldest group first. A group is the unit of removal: it is
        kept or removed entire, which is what keeps a tool call with its results and an assistant reply with the user
        message it answers.
    """
    # Steps belong to the current task, so they start after its anchor, or after the instructions when the
    # conversation has no user message to anchor on.
    step_start = (task_index + 1) if task_index is not None else system_end
    step_groups = _index_groups(messages=messages, spans=_agent_step_spans(messages=messages, start=step_start))

    # An earlier compaction's note is left out of its turn, so keeping the turn folds that note into the note this
    # compaction leaves behind.
    historical_end = task_index if task_index is not None else system_end
    historical_groups = _index_groups(
        messages=messages,
        spans=_historical_turn_spans(messages=messages, start=system_end, end=historical_end),
        skip_compaction_notes=True,
    )
    return historical_groups, step_groups


def _first_group_to_keep(
    messages: list[ChatMessage], groups: list[list[int]], available_tokens: int, token_counter: TokenCounter
) -> int:
    """
    Return the position in `groups` to start keeping from.

    Groups are added up from newest to oldest and counting stops at the first group that does not fit.

    :param messages: The full conversation containing the messages referenced by `groups`.
    :param groups: Ordered index groups to choose from, the oldest group first is at position 0.
    :param available_tokens: The token budget available for these groups.
    :param token_counter: The `TokenCounter` used to measure each group.
    :returns: The position in `groups` to start keeping from: `len(groups)` when nothing fits, `0` when it all fits.
    """
    # We count backwards so first_kept starts such that nothing would be kept
    first_kept = len(groups)
    while first_kept > 0:
        # Calculate the tokens consumed by the group that would be kept next
        group_tokens = token_counter.count(messages=_messages_at(messages=messages, indices=groups[first_kept - 1]))
        # If this group does not fit, we stop
        if group_tokens > available_tokens:
            break
        available_tokens -= group_tokens
        first_kept -= 1
    return first_kept


def _first_turn_and_step_to_keep(
    messages: list[ChatMessage],
    historical_groups: list[list[int]],
    step_groups: list[list[int]],
    available_tokens: int,
    token_counter: TokenCounter,
    min_keep_steps: int,
) -> tuple[int, int]:
    """
    Return which historical turn and which Agent step of the current task to start keeping from.

    :param messages: The full conversation containing the messages referenced by both group lists.
    :param historical_groups: Index groups for the complete historical turns preceding the current task, oldest first.
    :param step_groups: Index groups for the current task's Agent steps, oldest first.
    :param available_tokens: The token budget left once the protected context is paid for.
    :param token_counter: The `TokenCounter` used to measure the groups.
    :param min_keep_steps: The fewest recent Agent steps to keep, even when they exceed the budget.
    :returns: The position in `historical_groups` and the position in `step_groups` to start keeping from. Either is the
        length of its list when nothing from it is kept.
    """
    current_task_tokens = token_counter.count(
        messages=_messages_at(messages=messages, indices=_flatten(groups=step_groups))
    )
    if current_task_tokens > available_tokens:
        # The current task alone overruns the budget, so every historical turn is dropped and the current task's own
        # oldest steps trimmed until what remains fits.
        first_kept_step = _first_group_to_keep(
            messages=messages, groups=step_groups, available_tokens=available_tokens, token_counter=token_counter
        )
        # The newest steps are kept regardless of the budget.
        return len(historical_groups), min(first_kept_step, max(len(step_groups) - min_keep_steps, 0))

    # The entire current task fits, so every step stays and the rest of the budget goes on the newest turns that fit.
    first_kept_turn = _first_group_to_keep(
        messages=messages,
        groups=historical_groups,
        available_tokens=available_tokens - current_task_tokens,
        token_counter=token_counter,
    )
    return first_kept_turn, 0


def _task_and_step_split(
    messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, min_keep_steps: int
) -> tuple[list[ChatMessage], int, list[ChatMessage]]:
    """
    Split a conversation into the messages to keep and the messages to remove.

    Leading system messages and the latest real user message are always kept. Historical turns are kept in full when
    they fit. If the current task itself exceeds the available budget, its oldest Agent steps are removed one at a time
    while keeping each assistant message together with its tool results.

    :param messages: The full conversation to split, ordered oldest to newest.
    :param target_tokens: The token budget for the messages that are kept.
    :param token_counter: The `TokenCounter` used to decide which historical turns and current-task steps fit.
    :param min_keep_steps: The fewest recent current-task Agent steps to keep, even if they exceed the target token
        budget.
    :returns: A tuple containing:

        1. The messages to keep, ordered oldest to newest.
        2. The position in that list where an omission note belongs, which is where the removed messages used to sit:
           directly after the leading system messages when only historical turns were removed, and directly after the
           user message anchoring the current task when the task's own steps were removed.
        3. Every message selected for removal.
    """
    # The landmarks the split is built around: the Agent instructions, and the user message anchoring the current task.
    system_end = _leading_system_end(messages=messages)
    task_index = _latest_user_index(messages=messages)
    task_indices = [task_index] if task_index is not None else []

    # The two stretches compaction may remove. A group is the unit of removal, so a turn or a step is never split.
    historical_groups, step_groups = _removable_groups(messages=messages, system_end=system_end, task_index=task_index)

    # The instructions and the current task are never removed, so they come off the budget first.
    protected = _messages_at(messages=messages, indices=[*range(system_end), *task_indices])
    first_kept_turn, first_kept_step = _first_turn_and_step_to_keep(
        messages=messages,
        historical_groups=historical_groups,
        step_groups=step_groups,
        available_tokens=target_tokens - token_counter.count(messages=protected),
        token_counter=token_counter,
        min_keep_steps=min_keep_steps,
    )

    # What survives, laid out in conversation order: the instructions, the turns that fit, the task, then its steps.
    kept_turn_indices = _flatten(groups=historical_groups[first_kept_turn:])
    kept_step_indices = _flatten(groups=step_groups[first_kept_step:])
    kept_indices = [*range(system_end), *kept_turn_indices, *task_indices, *kept_step_indices]

    # The note stands in for what was dropped, so it goes where those messages used to sit. Either right after the
    # instructions when the historical turns were trimmed, or right after the task anchor when its own steps were.
    note_index = system_end if first_kept_step == 0 else system_end + len(kept_turn_indices) + len(task_indices)
    return (
        _messages_at(messages=messages, indices=kept_indices),
        note_index,
        _messages_except(messages=messages, indices=kept_indices),
    )


@_experimental
class SlidingWindowCompactor(Compactor):
    """
    Keeps the Agent's instructions, current task, and as much complete recent conversation as the target allows.

    Leading system messages and the latest user message are protected. Historical turns are kept in full when they fit,
    and the current task's history is kept in complete Agent steps, where a step is an assistant message together
    with all immediately following tool results.

    An `omission_note` is left where the removed messages used to sit: directly after the leading system messages when
    only historical turns were removed, and directly after the latest user message when the current task's own steps
    were removed. Only one note is ever present, since a later compaction folds an earlier note into its replacement.

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
        :param target_tokens: The size the kept conversation should come in under.
        :param token_counter: The `TokenCounter` to measure messages with.
        :returns: The conversation that survived, with an omission note if configured standing where the removed
            messages used to sit; or None when there is nothing to remove.
        """
        if token_counter.count(messages=messages) <= target_tokens:
            return None
        kept, note_index, removable = _task_and_step_split(
            messages=messages,
            target_tokens=target_tokens,
            token_counter=token_counter,
            min_keep_steps=self.min_keep_steps,
        )
        if not removable:
            return None
        if not self.omission_note:
            return kept

        # We prefer user over system since not all providers support multiple system messages
        note = ChatMessage.from_user(
            # `replace` rather than `format`, so a note carrying braces of its own cannot raise.
            self.omission_note.replace(_NUM_REMOVED_PLACEHOLDER, str(len(removable))),
            meta={
                _COMPACTION_META_KEY: {
                    "strategy": _STRATEGY,
                    "removed_messages": len(removable),
                    "kept_messages": len(kept),
                }
            },
        )
        return [*kept[:note_index], note, *kept[note_index:]]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(self, min_keep_steps=self.min_keep_steps, omission_note=self.omission_note)
