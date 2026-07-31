# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.token_counters import TokenCounter

# Meta key marking a message that a compactor produced. Its value records which strategy ran.
_COMPACTION_META_KEY = "context_compaction"


def _last_assistant_index(messages: list[ChatMessage]) -> int:
    """Return the index of the last assistant message, or -1 if none exists."""
    for index in reversed(range(len(messages))):
        if messages[index].is_from(role=ChatRole.ASSISTANT):
            return index
    return -1


def _estimated_context_tokens(messages: list[ChatMessage], context_tokens: int, token_counter: TokenCounter) -> int:
    """
    Estimate the size of the whole conversation.

    :param messages: The conversation, oldest to newest.
    :param context_tokens: The `context_tokens` state key which is computed using the provider's own token counting.
    :param token_counter: The counter to measure the unaccounted messages with.
    :returns: The estimated total token count.
    """
    # Nothing sent yet, or a generator that reports no usage, so count everything.
    if context_tokens == 0:
        return token_counter.count(messages=messages)
    # Only need to estimate the tool result messages after the last assistant message
    tool_result_messages = messages[_last_assistant_index(messages) + 1 :]
    return context_tokens + token_counter.count(messages=tool_result_messages)


def _compaction_split(
    messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter, min_keep_messages: int
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """
    Split the conversation into what compaction must keep and the block it may remove.

    :param messages: The conversation, oldest to newest.
    :param target_tokens: The size the conversation should come in under once the block is removed.
    :param token_counter: The counter to measure messages with.
    :param min_keep_messages: The fewest recent messages to keep, even when the target cannot afford them.
    :returns: The leading system messages, the removable block, and the retained window. The block is empty when there
        is nothing to remove.
    """
    # Protect only the *leading* run of system messages. A system message further along can be removed even ones added
    # by a compactor.
    start = 0
    while (
        start < len(messages)
        and messages[start].is_from(role=ChatRole.SYSTEM)
        and _COMPACTION_META_KEY not in messages[start].meta
    ):
        start += 1

    # We always keep the system messages so subtract its cost from the target to see what remains for the window.
    remaining_tokens = target_tokens - token_counter.count(messages=messages[:start])
    end = len(messages)
    while end > start:
        cost = token_counter.count(messages=[messages[end - 1]])
        if cost > remaining_tokens:
            break
        remaining_tokens -= cost
        end -= 1

    # Hold on to a few recent messages even when the target cannot pay for them, so the Agent keeps enough to carry on.
    end = min(end, max(len(messages) - min_keep_messages, start))

    # Walk the window's start back off any tool result, so it keeps the assistant message holding the matching call:
    # chat-completion APIs reject a tool result whose call is missing.
    while start < end < len(messages) and messages[end].tool_call_result is not None:
        end -= 1
    return messages[:start], messages[start:end], messages[end:]
