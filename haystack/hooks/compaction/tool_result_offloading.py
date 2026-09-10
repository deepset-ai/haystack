# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _agent_step_spans
from haystack.hooks.tool_result_offloading.types import ToolResultStore
from haystack.hooks.tool_result_offloading.utils import (
    _OFFLOADED_META_KEY,
    _offloadable_content_blocks,
    _offloaded_message,
)
from haystack.token_counters import TokenCounter
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental


@_experimental
class ToolResultOffloadCompactor(Compactor):
    """
    Writes older tool results to a `ToolResultStore`, leaving a compact reference in the conversation.

    Like `ToolResultPruningCompactor` this rewrites tool results in place, so every tool call keeps its matching
    result. The difference is that the full output survives: the model is left a reference it can read back with a
    tool scoped to the same store, instead of a placeholder telling it to run the tool again.

    Because `CompactionHook` only calls a compactor once the conversation crosses its threshold, fresh tool output
    goes to the model directly and only starts being offloaded under context pressure. Use `ToolResultOffloadHook`
    instead when a tool's output should always be offloaded, however short the run.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, ToolResultOffloadCompactor
    from haystack.hooks.tool_result_offloading import FileSystemToolResultStore

    hook = CompactionHook(
        compactor=ToolResultOffloadCompactor(store=FileSystemToolResultStore(root="tool_results")),
        context_window=400_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search, read_offloaded_result],
        hooks={"before_llm": [hook]},
    )
    ```

    A compactor takes no per-run context, so unlike `ToolResultOffloadHook` it cannot be given a store per run. In a
    multi-user server, build a compactor (and its hook) per run with that run's own isolated store, so concurrent
    users never read each other's offloaded results.

    Whether an image or file result is worth offloading depends on the `TokenCounter` in use: a local counter measures
    non-text content as a short stand-in, so offloading it frees nothing measurable and the result stays in context.
    A counter that measures what the provider actually charges for, such as `OpenAITokenCounter`, reports the real
    size and those results are offloaded like any other.
    """

    def __init__(
        self, store: ToolResultStore, *, min_keep_steps: int = 1, min_tokens: int = 200, preview_chars: int = 200
    ) -> None:
        """
        Initialize the compactor with its store and the rules deciding which results it offloads.

        :param store: Where offloaded results are written. A store that sets `supports_binary_content` also receives
            image and file results; with a text-only store those stay in the conversation.
        :param min_keep_steps: The minimum number of recent tool-calling Agent steps whose results remain untouched,
            even when they exceed the target. Must be at least 1, which ensures the current result batch remains intact
            until the model has acted on it.
        :param min_tokens: Only offload tool-result messages that use more than this many tokens. Small results cost
            little to keep and their references would save almost nothing.
        :param preview_chars: Number of leading characters of each offloaded text to include in the reference left in
            the conversation, so the model knows roughly what was offloaded. Image and file results are described by
            their MIME type and size instead.
        :raises ValueError: If `min_keep_steps` is less than 1, or `min_tokens` or `preview_chars` is negative.
        """
        if min_keep_steps < 1:
            raise ValueError(
                f"`min_keep_steps` must be at least 1, got {min_keep_steps}. The most recent tool-calling step "
                f"contains results the model may still need."
            )
        if min_tokens < 0:
            raise ValueError(f"`min_tokens` must be at least 0, got {min_tokens}.")
        if preview_chars < 0:
            raise ValueError(f"`preview_chars` must be at least 0, got {preview_chars}.")
        self.store = store
        self.min_keep_steps = min_keep_steps
        self.min_tokens = min_tokens
        self.preview_chars = preview_chars

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Replace the content of offloadable tool results with a reference to the stored result.

        Results are considered oldest first and offloading stops as soon as the conversation reaches `target_tokens`.
        This keeps as much output as possible directly in context. Results from the most recent `min_keep_steps`
        tool-calling Agent steps are never considered, even when the target cannot otherwise be reached. After
        measuring the initial conversation, the running total is updated with per-result token deltas to avoid
        repeatedly counting the full context.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results offloaded, or None when no result was offloadable.
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
        # Iterate oldest-first so we can stop at the target while leaving as much recent output in context as possible.
        for index, message in enumerate(messages):
            if message.tool_call_result is None or index in protected_positions:
                continue
            replacement = self._offload(message=message, index=index, token_counter=token_counter)
            if replacement is None:
                continue
            offloaded, saved_tokens = replacement

            # Update the compacted list and the running token count
            compacted[index] = offloaded
            current_tokens -= saved_tokens
            changed = True

            # Once we reach the target, stop offloading to keep as much recent output in context as possible.
            if current_tokens <= target_tokens:
                break

        # If no candidates were offloaded, return None to indicate no change.
        return compacted if changed else None

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Asynchronous version of `compact`, running the store writes in a thread so the event loop is not blocked.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results offloaded, or None when no result was offloadable.
        """
        return await asyncio.to_thread(
            self.compact, messages=messages, target_tokens=target_tokens, token_counter=token_counter
        )

    def _offload(self, message: ChatMessage, index: int, token_counter: TokenCounter) -> tuple[ChatMessage, int] | None:
        """
        Write one tool result to the store and report how many tokens its reference saves.

        The result's `origin` is carried over so the message keeps pointing at the tool call it answers, and its error
        flag is preserved.

        :param message: The tool-result message to consider.
        :param index: The message's position in the conversation, used to keep its store key unique.
        :param token_counter: The `TokenCounter` used to determine whether the result exceeds `min_tokens`.
        :returns: The offloaded message and the number of tokens it saves, or None when the result stays in context.
        """
        result = message.tool_call_result
        # Only successful tool output is offloaded - never errors, a result another compactor already rewrote, or one
        # that is already offloaded, whose content is the reference the model needs to read it back.
        if (
            result is None
            or result.error
            or message.meta.get(_OFFLOADED_META_KEY)
            or _COMPACTION_META_KEY in message.meta
        ):
            return None

        original_tokens = token_counter.count(messages=[message])
        # If the result is small enough, leave it alone.
        if original_tokens <= self.min_tokens:
            return None

        # An empty result, or image or file content a text-only store cannot take, stays in context.
        content_blocks = _offloadable_content_blocks(result=result, store=self.store)
        if content_blocks is None:
            return None

        # A tool call id is unique within a run, so it keeps results from different tools and steps from colliding; an
        # id-less call falls back to the message's position, which is unique within the conversation.
        offloaded = _offloaded_message(
            message=message,
            content_blocks=content_blocks,
            store=self.store,
            key_prefix=f"compacted_{result.origin.tool_name}_{result.origin.id or f'call{index}'}",
            preview_chars=self.preview_chars,
            additional_meta={
                _COMPACTION_META_KEY: {"strategy": "tool_result_offloading", "original_tokens": original_tokens}
            },
        )
        saved_tokens = original_tokens - token_counter.count(messages=[offloaded])
        # A reference to a barely-larger-than-`min_tokens` result can cost more than the result itself; only return
        # replacements that reduce context. The written store entry is then left unreferenced.
        return (offloaded, saved_tokens) if saved_tokens > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the compactor, including its store.

        :returns: A dictionary representation of the compactor.
        """
        return default_to_dict(
            self,
            store=self.store.to_dict(),
            min_keep_steps=self.min_keep_steps,
            min_tokens=self.min_tokens,
            preview_chars=self.preview_chars,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultOffloadCompactor":
        """
        Deserialize the compactor, reconstructing its store.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `ToolResultOffloadCompactor`.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("store") is not None:
            deserialize_component_inplace(init_params, key="store")
        return default_from_dict(cls, data)
