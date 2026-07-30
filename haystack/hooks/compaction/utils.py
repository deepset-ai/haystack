# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.token_counters import TokenCounter

# Meta key marking a message that a compactor produced. Its value records which strategy ran.
_COMPACTION_META_KEY = "context_compaction"


def _last_assistant_end(messages: list[ChatMessage]) -> int:
    """The index just past the most recent assistant message, or 0 when there is none."""
    # A generator's usage covers the request it was sent plus the reply it produced, and that reply is the last
    # assistant message - so this is the boundary of what has already been counted for us.
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].is_from(ChatRole.ASSISTANT):
            return index + 1
    return 0


def _estimated_context_tokens(messages: list[ChatMessage], context_tokens: int, counter: TokenCounter) -> int:
    """
    Estimate the size of the whole conversation.

    :param messages: The conversation, oldest to newest.
    :param context_tokens: The `context_tokens` state key which is computed using the provider's own token counting.
    :param counter: The counter to measure the unaccounted messages with.
    :returns: The estimated total token count.
    """
    # Nothing sent yet, or a generator that reports no usage: there is no anchor, so count the whole conversation.
    if context_tokens == 0:
        return counter.count(messages)
    # We only estimate the messages that arrived after the last assistant message to keep the count as accurate as
    # possible.
    return context_tokens + counter.count(messages[_last_assistant_end(messages) :])


def _compaction_bounds(
    messages: list[ChatMessage], target_tokens: int, counter: TokenCounter, min_keep_messages: int
) -> tuple[int, int]:
    """
    Return where the block of messages that compaction may remove starts and ends.

    The block runs from `start` up to but not including `end`, so `messages[start:end]` is what may go. Before it is the
    leading run of system messages and after it is the recent window, both of which are kept. The window takes as much
    recent history as `target_tokens` affords.

    :param messages: The conversation, oldest to newest.
    :param target_tokens: The size the conversation should come in under once the block is removed.
    :param counter: The counter to measure messages with.
    :param min_keep_messages: The fewest recent messages to keep, even when the target cannot afford them.
    :returns: The `(start, end)` indices of the removable block; `end - start` is `0` when there is nothing to remove.
    """
    # Protect only the *leading* run of system messages: the Agent's system prompt and anything a hook prepended. A
    # system message further along is removed like any other. The marker check ends the run at a message a compactor
    # produced, since that is removable.
    start = 0
    while (
        start < len(messages)
        and messages[start].is_from(ChatRole.SYSTEM)
        and _COMPACTION_META_KEY not in messages[start].meta
    ):
        start += 1

    # We always keep the system messages so subtract its cost from the target to see what remains for the window.
    remaining = target_tokens - counter.count(messages[:start])
    end = len(messages)
    while end > start:
        cost = counter.count([messages[end - 1]])
        if cost > remaining:
            break
        remaining -= cost
        end -= 1

    # Hold on to a few recent messages even when the target cannot pay for them, so the Agent keeps enough to carry on.
    end = min(end, max(len(messages) - min_keep_messages, start))

    # Walk the window's start back off any tool result, so it keeps the assistant message holding the matching call:
    # chat-completion APIs reject a tool result whose call is missing. One pass steps over a whole parallel batch.
    while start < end < len(messages) and messages[end].tool_call_result is not None:
        end -= 1
    return start, end
