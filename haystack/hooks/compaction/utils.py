# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.token_counters import TokenCounter

# Meta key marking a message that a compactor produced. Its value records which strategy ran.
_COMPACTION_META_KEY = "context_compaction"


def _leading_system_end(messages: list[ChatMessage]) -> int:
    """Return the end of the leading system-message block, excluding system messages created by compaction."""
    for index, message in enumerate(messages):
        if not message.is_from(role=ChatRole.SYSTEM) or _COMPACTION_META_KEY in message.meta:
            return index
    return len(messages)


def _latest_user_index(messages: list[ChatMessage]) -> int | None:
    """Return the latest user message not produced by compaction."""
    for index in reversed(range(len(messages))):
        message = messages[index]
        if message.is_from(role=ChatRole.USER) and _COMPACTION_META_KEY not in message.meta:
            return index
    return None


def _messages_at(messages: list[ChatMessage], indices: list[int]) -> list[ChatMessage]:
    """Return the messages at the given indices, in the order the indices are given."""
    return [messages[index] for index in indices]


def _messages_except(messages: list[ChatMessage], indices: list[int]) -> list[ChatMessage]:
    """Return the messages the given indices leave out, in conversation order."""
    left_out = set(indices)
    return [message for index, message in enumerate(messages) if index not in left_out]


def _is_compaction_message(message: ChatMessage, strategy: str, role: ChatRole | None = None) -> bool:
    """Return whether a message was produced by a compaction strategy and optionally has the requested role."""
    marker = message.meta.get(_COMPACTION_META_KEY)
    has_role = role is None or message.is_from(role=role)
    return has_role and isinstance(marker, dict) and marker.get("strategy") == strategy


def _last_assistant_index(messages: list[ChatMessage]) -> int:
    """Return the index of the last assistant message, or -1 if none exists."""
    for index in reversed(range(len(messages))):
        if messages[index].is_from(role=ChatRole.ASSISTANT):
            return index
    return -1


def _agent_step_spans(messages: list[ChatMessage], start: int) -> list[tuple[int, int]]:
    """
    Return spans containing an assistant message and all immediately following tool results.

    :param messages: The conversation to analyze, oldest to newest.
    :param start: The index to start searching for steps from. We recommend starting after the latest user message, or
        after the leading system messages when there is no user message. Otherwise, the returned spans may include
        steps that are not part of the current task.
    :returns: A list of spans, where each span is a tuple of (start_index, end_index) and end_index is exclusive.
    """
    # e.g. a span of (2, 5) means messages[2:5] are part of the same step
    spans: list[tuple[int, int]] = []
    index = start
    while index < len(messages):
        # Only assistant messages can start a step; skip any other messages.
        if not messages[index].is_from(role=ChatRole.ASSISTANT):
            index += 1
            continue
        # Find the end of the step by looking for the first message that is not a tool result.
        end = index + 1
        while end < len(messages) and messages[end].tool_call_results:
            end += 1
        # Record the span and continue searching for the next step.
        spans.append((index, end))
        # Reset the index to the end of the current step to avoid overlapping spans.
        index = end
    return spans


def _current_agent_step_groups(messages: list[ChatMessage], system_end: int, task_index: int | None) -> list[list[int]]:
    """
    Return message-index groups for complete Agent steps belonging to the current task.

    :param messages: The conversation to analyze, ordered oldest to newest.
    :param system_end: The end of the leading system-message block. Steps are looked for from here when the
        conversation has no user message to anchor on.
    :param task_index: The index of the user message anchoring the current task, or None when there is none. Steps are
        looked for from the message after it.
    :returns: One group of message indices per step, ordered oldest step first. A step is an assistant message and all
        immediately following tool results, so a group holds a tool call together with its results.
    """
    step_start = task_index + 1 if task_index is not None else system_end
    return [list(range(start, end)) for start, end in _agent_step_spans(messages=messages, start=step_start)]


def _historical_turn_spans(messages: list[ChatMessage], start: int, end: int) -> list[tuple[int, int]]:
    """
    Return spans for complete real-user turns in a bounded section of conversation history.

    Each turn begins with a user message not created by compaction and continues up to the next such message.
    """
    user_indices = [
        index
        for index in range(start, end)
        if messages[index].is_from(role=ChatRole.USER) and _COMPACTION_META_KEY not in messages[index].meta
    ]
    return [
        (index, user_indices[position + 1] if position + 1 < len(user_indices) else end)
        for position, index in enumerate(user_indices)
    ]


def _historical_turn_groups(messages: list[ChatMessage], system_end: int, task_index: int | None) -> list[list[int]]:
    """
    Return message-index groups for complete historical turns preceding the current task.

    :param messages: The conversation to analyze, ordered oldest to newest.
    :param system_end: The end of the leading system-message block, where the history begins.
    :param task_index: The index of the user message anchoring the current task, where the history ends. None when
        there is no such message, in which case the history is empty and everything belongs to the current task.
    :returns: One group of message indices per turn, ordered oldest turn first. A turn is a user message that
        compaction did not produce, together with everything that follows it up to the next one.
    """
    historical_end = task_index if task_index is not None else system_end
    return [
        list(range(start, end))
        for start, end in _historical_turn_spans(messages=messages, start=system_end, end=historical_end)
    ]


def _replace_tool_results_until_target(
    messages: list[ChatMessage],
    *,
    target_tokens: int,
    token_counter: TokenCounter,
    min_keep_steps: int,
    replace: Callable[[ChatMessage, int], tuple[ChatMessage, int] | None],
) -> list[ChatMessage] | None:
    """
    Replace tool results oldest first until the conversation fits within `target_tokens`.

    Results from the most recent `min_keep_steps` tool-calling Agent steps are never passed to `replace`. After
    measuring the initial conversation, the running total is updated with each replacement's saved tokens to avoid
    recounting the full context.

    :param messages: The conversation to compact, oldest to newest.
    :param target_tokens: The token count the compacted conversation should fit within.
    :param token_counter: The `TokenCounter` used to measure the initial conversation.
    :param min_keep_steps: Number of most recent tool-calling Agent steps whose results are left unchanged.
    :param replace: Called with a tool-result message and its position. Returns the replacement and the tokens it
        saves, or None to leave the message unchanged.
    :returns: The conversation with replaced tool results, or None when nothing was replaced.
    """
    current_tokens = token_counter.count(messages=messages)
    if current_tokens <= target_tokens:
        return None

    # Filter the shared Agent-step spans to tool-calling steps; parallel results remain grouped in one span.
    result_steps = [
        list(range(start + 1, end))
        for start, end in _agent_step_spans(messages=messages, start=0)
        # An assistant-only span has no result to protect and must not consume one of `min_keep_steps`.
        if end > start + 1
    ]
    protected_positions = {position for step in result_steps[-min_keep_steps:] for position in step}

    # Replace entries in a new list so the caller-owned input list remains unchanged.
    compacted = list(messages)
    changed = False
    # Go oldest first and stop at the target, so as much recent output as possible stays in context.
    for index, message in enumerate(messages):
        if message.tool_call_result is None or index in protected_positions:
            continue
        replacement = replace(message, index)
        if replacement is None:
            continue
        compacted[index], saved_tokens = replacement
        current_tokens -= saved_tokens
        changed = True
        if current_tokens <= target_tokens:
            break

    return compacted if changed else None
