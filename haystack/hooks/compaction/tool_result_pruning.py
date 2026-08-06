# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _agent_step_spans
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
    from haystack.hooks.compaction import CompactionHook, ToolResultPruningCompactor

    hook = CompactionHook(
        compactor=ToolResultPruningCompactor(min_keep_steps=1),
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
        min_keep_steps: int = 1,
        min_tokens: int = 200,
        placeholder: str = _DEFAULT_PLACEHOLDER,
        skip_meta_keys: tuple[str, ...] = ("tool_result_offloaded",),
    ) -> None:
        """
        Initialize the compactor with the rules deciding which results it prunes.

        :param min_keep_steps: The minimum number of recent tool-calling Agent steps whose results remain untouched,
            even when they exceed the target. Must be at least 1, which ensures the current result batch remains intact
            until the model has acted on it.
        :param min_tokens: Only prune tool-result messages that use more than this many tokens. Small results cost
            little and are often the ones worth keeping.
        :param placeholder: The text left in place of a pruned result, replacing the built-in one. May contain
            `{tool_name}`, which is filled in with the name of the tool that produced the result.
        :param skip_meta_keys: Results whose `meta` contains any of these keys are left alone. The default covers
            results that a `ToolResultOffloadHook` already replaced with a reference to stored content: pruning one of
            those would destroy the reference the model needs to read it back.
        :raises ValueError: If `min_keep_steps` is less than 1 or `min_tokens` is negative.
        """
        if min_keep_steps < 1:
            raise ValueError(
                f"`min_keep_steps` must be at least 1, got {min_keep_steps}. The most recent tool-calling step "
                f"contains results the model may still need."
            )
        if min_tokens < 0:
            raise ValueError(f"`min_tokens` must be at least 0, got {min_tokens}.")
        self.min_keep_steps = min_keep_steps
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
        This keeps as much original output as possible. Results from the most recent `min_keep_steps` tool-calling
        Agent steps are never considered, even when the target cannot otherwise be reached. After measuring the initial
        conversation, the running total is updated with per-result token deltas to avoid repeatedly counting the full
        context.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results replaced, or None when no result was prunable.
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

        protected_positions = {position for step in result_steps[-self.min_keep_steps :] for position in step}

        # Replace entries in a new list so the caller-owned input list remains unchanged.
        compacted = list(messages)
        changed = False
        # Iterate oldest-first so we can stop at the target while leaving as much recent output intact as possible.
        for index, message in enumerate(messages):
            if message.tool_call_result is None or index in protected_positions:
                continue
            replacement = self._prune(message=message, token_counter=token_counter)
            if replacement is None:
                continue
            pruned, saved_tokens = replacement

            # Update the compacted list and the running token count
            compacted[index] = pruned
            current_tokens -= saved_tokens
            changed = True

            # Once we reach the target, stop pruning to preserve as much recent output as possible.
            if current_tokens <= target_tokens:
                break

        # If no candidates were pruned, return None to indicate no change.
        return compacted if changed else None

    def _prune(self, message: ChatMessage, token_counter: TokenCounter) -> tuple[ChatMessage, int] | None:
        """
        Rewrite one tool-result message with a placeholder, or return None to leave it as it is.

        The result's `origin` is carried over so the message keeps pointing at the tool call it answers, and its error
        flag is preserved.

        :param message: The tool-result message to consider.
        :param token_counter: The `TokenCounter` used to determine whether the result exceeds `min_tokens`.
        :returns: The rewritten message and number of tokens it saves, or None when this result is not prunable.
        """
        result = message.tool_call_result
        # Guard against messages that are not tool results or have already been compacted.
        if result is None or _COMPACTION_META_KEY in message.meta:
            return None

        # Guard against messages that have meta keys indicating they should be skipped.
        if any(key in message.meta for key in self.skip_meta_keys):
            return None

        original_tokens = token_counter.count(messages=[message])
        # If the result is small enough, leave it alone.
        if original_tokens <= self.min_tokens:
            return None

        # `replace` permits custom placeholders to contain unrelated braces, such as JSON examples.
        placeholder = self.placeholder.replace("{tool_name}", result.origin.tool_name)
        pruned = ChatMessage.from_tool(
            tool_result=placeholder,
            origin=result.origin,
            error=result.error,
            meta={
                **message.meta,
                _COMPACTION_META_KEY: {"strategy": "tool_result_pruning", "original_tokens": original_tokens},
            },
        )
        saved_tokens = original_tokens - token_counter.count(messages=[pruned])
        # A custom placeholder can be larger than the original result; only return replacements that reduce context.
        return (pruned, saved_tokens) if saved_tokens > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(
            self,
            min_keep_steps=self.min_keep_steps,
            min_tokens=self.min_tokens,
            placeholder=self.placeholder,
            skip_meta_keys=list(self.skip_meta_keys),
        )
