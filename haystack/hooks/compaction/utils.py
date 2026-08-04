# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.token_counters import TokenCounter
from haystack.tools import ToolsType

# Meta key marking a message that a compactor produced. Its value records which strategy ran.
_COMPACTION_META_KEY = "context_compaction"


def _last_assistant_index(messages: list[ChatMessage]) -> int:
    """Return the index of the last assistant message, or -1 if none exists."""
    for index in reversed(range(len(messages))):
        if messages[index].is_from(role=ChatRole.ASSISTANT):
            return index
    return -1


def _estimated_context_tokens(
    messages: list[ChatMessage], context_tokens: int, token_counter: TokenCounter, tools: ToolsType | None = None
) -> int:
    """
    Estimate the size of the whole conversation.

    :param messages: The conversation, oldest to newest.
    :param context_tokens: The `context_tokens` state key which is computed using the provider's own token counting.
    :param token_counter: The counter to measure the unaccounted messages with.
    :param tools: Tools whose schemas are sent alongside the messages. These are counted when provider usage is absent.
    :returns: The estimated total token count.
    """
    # Nothing sent yet, or a generator that reports no usage, so count everything.
    if context_tokens == 0:
        return token_counter.count(messages=messages, tools=tools)
    # Only need to estimate the tool result messages after the last assistant message
    tool_result_messages = messages[_last_assistant_index(messages=messages) + 1 :]
    return context_tokens + token_counter.count(messages=tool_result_messages)
