# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole

# Meta key marking a message that a compactor produced. Its value records which strategy ran and at which step.
_COMPACTION_META_KEY = "context_compaction"


def _compaction_bounds(messages: list[ChatMessage], keep_last_n: int) -> tuple[int, int]:
    """
    Return the half-open range of messages that compaction may remove.

    Before the range sits the leading block of system messages: the Agent's standing instructions, which stay in context
    however often the conversation is compacted. That block ends at the first system message carrying
    `_COMPACTION_META_KEY`, because an earlier compaction produced that one and the next compaction replaces it instead
    of accumulating beside it.

    After the range sits the tail of `keep_last_n` recent messages. The tail's start is moved backwards off any
    tool-result message: the matching tool call lives in the assistant message before it, which compaction is about to
    remove, and a tool result whose call is missing is rejected by chat-completion APIs. Moving backwards retains more
    than `keep_last_n` rather than fewer, and steps over a whole batch of parallel tool results in one pass. Nothing is
    checked inside the range itself, since it is removed as a block rather than reconnected to anything.

    :param messages: The conversation, oldest to newest.
    :param keep_last_n: How many trailing messages to retain, before the adjustment described above.
    :returns: The `(start, end)` indices of the removable range, where `end - start` is how many messages it covers and
        `0` means there is nothing to remove.
    """
    start = 0
    while (
        start < len(messages)
        and messages[start].is_from(ChatRole.SYSTEM)
        and _COMPACTION_META_KEY not in messages[start].meta
    ):
        start += 1

    end = max(len(messages) - keep_last_n, start)
    while end > start and messages[end].tool_call_result is not None:
        end -= 1
    return start, end
