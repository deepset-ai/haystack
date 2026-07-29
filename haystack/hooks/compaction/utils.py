# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole

# Meta key marking a message that a compactor produced. Its value records which strategy ran.
_COMPACTION_META_KEY = "context_compaction"


def _compaction_bounds(messages: list[ChatMessage], keep_last_n: int) -> tuple[int, int]:
    """
    Return where the block of messages that compaction may remove starts and ends.

    The block runs from `start` up to but not including `end`, so `messages[start:end]` is what may go. Before it is the
    leading run of system messages and after it is the recent window, both of which are kept.

    :param messages: The conversation, oldest to newest.
    :param keep_last_n: How many trailing messages to retain, before the adjustment made below.
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

    # Walk the tail's start back off any tool result, so the tail keeps the assistant message holding the matching call:
    # chat-completion APIs reject a tool result whose call is missing. One pass steps over a whole parallel batch.
    end = max(len(messages) - keep_last_n, start)
    while end > start and messages[end].tool_call_result is not None:
        end -= 1
    return start, end
