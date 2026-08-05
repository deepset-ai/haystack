# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY
from haystack.token_counters import TokenCounter
from haystack.utils.experimental import _experimental

_DEFAULT_PLACEHOLDER = "[Tool result removed to free up context. Call `{tool_name}` again if you need it.]"


@_experimental
class ToolResultPruningCompactor(Compactor):
    """
    Replaces the content of older tool results with a short placeholder, keeping the conversation's shape intact.

    Tool output usually dominates a long Agent run, and most of it stops being useful once the model has acted on it.
    This compactor rewrites those results in place rather than removing messages, so every tool call keeps its matching
    result and the model can see what it ran and re-run it if needed.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, ToolResultPruningCompactor

    hook = ContextCompactionHook(
        compactor=ToolResultPruningCompactor(min_keep_results=3),
        context_window=400_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
    )
    ```
    """

    def __init__(
        self,
        *,
        min_keep_results: int = 3,
        min_tokens: int = 200,
        placeholder: str = _DEFAULT_PLACEHOLDER,
        skip_meta_keys: tuple[str, ...] = ("tool_result_offloaded",),
    ) -> None:
        """
        Initialize the compactor with the rules deciding which results it prunes.

        :param min_keep_results: The minimum number of recent tool results to leave untouched, even when they exceed
            the target. More results remain intact when the target can be reached without pruning them. Must be at
            least 1. The entire trailing batch of results from the current Agent step is always kept, even when it is
            larger than this minimum, because the model has not acted on those results yet.
        :param min_tokens: Only prune tool-result messages that use more than this many tokens. Small results cost
            little and are often the ones worth keeping. Token-based sizing also accounts for image and file content.
        :param placeholder: The text left in place of a pruned result, replacing the built-in one. May contain
            `{tool_name}`, which is filled in with the name of the tool that produced the result.
        :param skip_meta_keys: Results whose `meta` contains any of these keys are left alone. The default covers
            results that a `ToolResultOffloadHook` already replaced with a reference to stored content: pruning one of
            those would destroy the reference the model needs to read it back.
        :raises ValueError: If `min_keep_results` is less than 1 or `min_tokens` is negative.
        """
        if min_keep_results < 1:
            raise ValueError(
                f"`min_keep_results` must be at least 1, got {min_keep_results}. The most recent tool result is "
                f"the one the model is about to act on."
            )
        if min_tokens < 0:
            raise ValueError(f"`min_tokens` must be at least 0, got {min_tokens}.")
        self.min_keep_results = min_keep_results
        self.min_tokens = min_tokens
        self.placeholder = placeholder
        # Normalized to a tuple so a round trip through `to_dict`, which has to emit a list, restores the same type.
        self.skip_meta_keys = tuple(skip_meta_keys)

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Replace the content of prunable tool results with a placeholder.

        Results are considered oldest first and pruning stops as soon as the conversation reaches `target_tokens`.
        This keeps as much original output as possible. The most recent `min_keep_results` results are never
        considered, even when the target cannot otherwise be reached. The trailing batch of tool results is also
        protected in full because all of those results belong to the current Agent step.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results replaced, or None when no result was prunable.
        """
        current_tokens = token_counter.count(messages)
        if current_tokens <= target_tokens:
            return None

        # Always protect trailing batch of results even when it is larger than `min_keep_results`, b/c the model has not
        # acted on those results yet.
        trailing_result_count = 0
        for message in reversed(messages):
            if message.tool_call_result is None:
                break
            trailing_result_count += 1
        # The number of results to protect is the larger of the configured minimum and the trailing batch.
        protected_result_count = max(self.min_keep_results, trailing_result_count)

        # Only consider results that are not in the protected set, and stop when the target is reached.
        result_positions = [index for index, message in enumerate(messages) if message.tool_call_result is not None]
        candidates = result_positions[:-protected_result_count]
        if not candidates:
            return None

        compacted = list(messages)
        changed = False
        for index in candidates:
            pruned = self._prune(messages[index], token_counter)
            if pruned is None:
                continue

            candidate = list(compacted)
            candidate[index] = pruned
            candidate_tokens = token_counter.count(candidate)
            # A custom placeholder can be larger than the result. Compactors must only return a shorter conversation,
            # so retain the original in that case.
            if candidate_tokens >= current_tokens:
                continue

            compacted = candidate
            current_tokens = candidate_tokens
            changed = True
            if current_tokens <= target_tokens:
                break
        return compacted if changed else None

    def _prune(self, message: ChatMessage, token_counter: TokenCounter) -> ChatMessage | None:
        """
        Rewrite one tool-result message with a placeholder, or return None to leave it as it is.

        The result's `origin` is carried over so the message keeps pointing at the tool call it answers, and its error
        flag is preserved. Errors are left alone: they are short and tell the model something it needs.

        :param message: The tool-result message to consider.
        :param token_counter: The `TokenCounter` used to determine whether the result exceeds `min_tokens`.
        :returns: The rewritten message, or None when this result is not pruned.
        """
        result = message.tool_call_result
        if result is None or result.error or _COMPACTION_META_KEY in message.meta:
            return None
        if any(key in message.meta for key in self.skip_meta_keys):
            return None

        original_tokens = token_counter.count([message])
        if original_tokens <= self.min_tokens:
            return None

        # `replace` permits custom placeholders to contain unrelated braces, such as JSON examples.
        placeholder = self.placeholder.replace("{tool_name}", result.origin.tool_name)
        return ChatMessage.from_tool(
            tool_result=placeholder,
            origin=result.origin,
            error=result.error,
            meta={
                **message.meta,
                _COMPACTION_META_KEY: {"strategy": "tool_result_pruning", "original_tokens": original_tokens},
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(
            self,
            min_keep_results=self.min_keep_results,
            min_tokens=self.min_tokens,
            placeholder=self.placeholder,
            skip_meta_keys=list(self.skip_meta_keys),
        )
