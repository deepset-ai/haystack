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
    Writes older tool results to a `ToolResultStore` and leaves a reference in their place.

    Every tool call keeps its matching result, and the model can read the full output back with a read tool you scope
    to the same store. `CompactionHook` only calls the compactor once the conversation reaches `compact_at`, so fresh
    output stays in context until then. Use `ToolResultOffloadHook` to offload a tool's output as soon as it arrives.

    <!-- test-ignore -->
    ```python
    from typing import Annotated

    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, ToolResultOffloadCompactor
    from haystack.hooks.tool_result_offloading import FileSystemToolResultStore
    from haystack.tools import tool

    store = FileSystemToolResultStore(root="tool_results")


    @tool
    def read_offloaded_result(path: Annotated[str, "Path of an offloaded tool result"]) -> str:
        '''Read back the full content of an offloaded tool result.'''
        content = store.read(path)
        if isinstance(content, bytes):
            return f"'{path}' holds {len(content)} bytes of binary content and cannot be read as text."
        return content


    hook = CompactionHook(compactor=ToolResultOffloadCompactor(store=store), context_window=400_000)
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search, read_offloaded_result],
        hooks={"before_llm": [hook]},
    )
    ```

    The compactor always writes to the store it was created with; unlike `ToolResultOffloadHook`, it does not read a
    per-run store from `hook_context`. In a multi-user server, create an Agent per run with its own compactor and
    store, so users never read each other's results.

    For results with images or files, we recommend passing a provider token counter such as `OpenAITokenCounter` to
    `CompactionHook`, which measures their real size. Its default `ApproximateTokenCounter` charges a flat
    `tokens_per_image` and `tokens_per_file` instead, which can undercount such results and keep them below
    `min_tokens`.
    """

    def __init__(
        self, store: ToolResultStore, *, min_keep_steps: int = 1, min_tokens: int = 200, preview_chars: int = 200
    ) -> None:
        """
        Initialize the compactor.

        :param store: Where offloaded results are written. Image and file results are only written to a store that sets
            `supports_binary_content`; otherwise they stay in the conversation.
        :param min_keep_steps: Number of most recent tool-calling Agent steps whose results are never offloaded, even
            when the target is missed. Must be at least 1, so the model always sees the latest results.
        :param min_tokens: Only offload tool-result messages larger than this many tokens.
        :param preview_chars: Number of leading characters of each offloaded text kept in its reference. Image and file
            results are described by MIME type and size instead.
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
        Offload tool results to the store, oldest first, until the conversation fits within `target_tokens`.

        Results from the most recent `min_keep_steps` tool-calling Agent steps are never offloaded.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The token count the compacted conversation should fit within.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results replaced by references, or None when nothing was offloaded.
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
        # Go oldest first and stop at the target, so as much recent output as possible stays in context.
        for index, message in enumerate(messages):
            if message.tool_call_result is None or index in protected_positions:
                continue
            replacement = self._offload(message=message, index=index, token_counter=token_counter)
            if replacement is None:
                continue
            offloaded, saved_tokens = replacement
            compacted[index] = offloaded
            current_tokens -= saved_tokens
            changed = True
            if current_tokens <= target_tokens:
                break

        return compacted if changed else None

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Run `compact` in a thread so store writes and token counting do not block the event loop.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The token count the compacted conversation should fit within.
        :param token_counter: The `TokenCounter` used to measure the conversation before and after each replacement.
        :returns: The conversation with older tool results replaced by references, or None when nothing was offloaded.
        """
        return await asyncio.to_thread(
            self.compact, messages=messages, target_tokens=target_tokens, token_counter=token_counter
        )

    def _offload(self, message: ChatMessage, index: int, token_counter: TokenCounter) -> tuple[ChatMessage, int] | None:
        """
        Write one tool result to the store and report how many tokens its reference saves.

        :param message: The tool-result message to consider.
        :param index: The message's position in the conversation, used in the store key when the tool call has no id.
        :param token_counter: The `TokenCounter` that measures the result and its reference.
        :returns: The offloaded message and the number of tokens it saves, or None when the result stays in context.
        """
        result = message.tool_call_result
        # Skip errors, results another compactor already rewrote, and results that are already a reference.
        if (
            result is None
            or result.error
            or message.meta.get(_OFFLOADED_META_KEY)
            or _COMPACTION_META_KEY in message.meta
        ):
            return None

        original_tokens = token_counter.count(messages=[message])
        if original_tokens <= self.min_tokens:
            return None

        # An empty result, or image or file content a text-only store cannot take, stays in context.
        content_blocks = _offloadable_content_blocks(result=result, store=self.store)
        if content_blocks is None:
            return None

        # The tool call id keeps store keys unique; an id-less call falls back to the message's position.
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
        # A reference can cost more than a small result. That replacement is dropped and its store entry left unused.
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
