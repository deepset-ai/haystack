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
    Split a conversation into:

    1. Protected leading system messages.
    2. Older messages that may be compacted.
    3. Recent messages that must be retained.
    """
    # Protect the initial run of ordinary system messages. System messages added by a previous compaction are not
    # protected.
    protected_end = 0
    while protected_end < len(messages):
        message = messages[protected_end]
        is_protected_system_message = message.is_from(ChatRole.SYSTEM) and _COMPACTION_META_KEY not in message.meta
        if not is_protected_system_message:
            break
        protected_end += 1

    protected_messages = messages[:protected_end]
    available_tokens = target_tokens - token_counter.count(messages=protected_messages)

    # Starting from the newest message, find how much recent history fits.
    retained_start = len(messages)
    while retained_start > protected_end:
        next_message = messages[retained_start - 1]
        message_tokens = token_counter.count(messages=[next_message])
        if message_tokens > available_tokens:
            break
        available_tokens -= message_tokens
        retained_start -= 1

    # Always retain at least the requested number of recent messages, even if they exceed the token target.
    earliest_allowed_start = max(len(messages) - min_keep_messages, protected_end)
    retained_start = min(retained_start, earliest_allowed_start)

    # A tool result must not be retained without the assistant message containing its corresponding tool call.
    while protected_end < retained_start < len(messages) and messages[retained_start].tool_call_result is not None:
        retained_start -= 1

    compactable_messages = messages[protected_end:retained_start]
    retained_messages = messages[retained_start:]
    return protected_messages, compactable_messages, retained_messages
