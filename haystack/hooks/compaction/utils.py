# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage, ChatRole

# Meta key marking a message that a compactor produced (a summary, an omission note, or a rewritten tool result). Its
# value records which strategy ran and at which step. Compactors both write and read it: it keeps a rewritten message
# from being rewritten again, and it tells `_preserved_prefix_end` that a leading system message is a previous summary
# rather than part of the conversation's permanent system prefix.
_COMPACTION_META_KEY = "context_compaction"


def _preserved_prefix_end(messages: list[ChatMessage]) -> int:
    """
    Return the length of the leading block of system messages that compaction must never touch.

    These are the Agent's standing instructions, so they stay in context however often the conversation is compacted.
    A leading system message carrying `_COMPACTION_META_KEY` is excluded: it is a summary an earlier compaction
    inserted, and treating it as permanent prefix would leave one summary behind per compaction instead of folding
    each into the next.

    :param messages: The conversation, oldest to newest.
    :returns: The index of the first message that may be compacted.
    """
    index = 0
    while (
        index < len(messages)
        and messages[index].is_from(ChatRole.SYSTEM)
        and _COMPACTION_META_KEY not in messages[index].meta
    ):
        index += 1
    return index


def _safe_cut_index(messages: list[ChatMessage], prefix_end: int, keep_last_n: int) -> int:
    """
    Return the index where the retained tail starts, moved to a boundary that leaves a valid conversation.

    The tail must not begin with a tool-result message: its matching tool call lives in the assistant message before
    it, which compaction is about to remove, and a tool result whose call is missing is rejected by chat-completion
    APIs. Walking the boundary backwards onto that assistant message retains more than `keep_last_n` rather than
    fewer, and steps over a whole batch of parallel tool results in one pass.

    Nothing is checked about the messages before the boundary, because they are removed as a block rather than
    reconnected to anything.

    :param messages: The conversation, oldest to newest.
    :param prefix_end: The index the compactable region starts at, from `_preserved_prefix_end`. The boundary is never
        moved before it.
    :param keep_last_n: How many trailing messages to retain, before the adjustment.
    :returns: The index of the first retained message. Equal to `prefix_end` when there is nothing to remove.
    """
    cut = max(len(messages) - keep_last_n, prefix_end)
    while cut > prefix_end and messages[cut].tool_call_result is not None:
        cut -= 1
    return cut
