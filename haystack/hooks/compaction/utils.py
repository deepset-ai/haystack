# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole

# Meta key marking a message that a compactor produced. Its value records which strategy ran and at which step.
_COMPACTION_META_KEY = "context_compaction"


def _compaction_bounds(messages: list[ChatMessage], keep_last_n: int) -> tuple[int, int]:
    """
    Return the half-open range of messages that compaction may remove.

    The range starts after the leading block of system messages, the Agent's standing instructions, which stay in
    context however often the conversation is compacted. That block ends at the first system message carrying
    `_COMPACTION_META_KEY`, since an earlier compaction produced that one and the next compaction replaces it.

    The range ends before the tail of `keep_last_n` recent messages, moved back off any tool-result message so that the
    tail keeps the assistant message holding the matching call - a tool result whose call is missing is rejected by
    chat-completion APIs. Moving back also steps over a whole batch of parallel tool results in one pass.

    :param messages: The conversation, oldest to newest.
    :param keep_last_n: How many trailing messages to retain, before the adjustment described above.
    :returns: The `(start, end)` indices of the removable range; `end - start` is `0` when there is nothing to remove.
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
