# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
from typing import Any

from haystack import logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _COMPACTION_META_KEY, _agent_step_spans
from haystack.hooks.tool_result_offloading.hooks import (
    _OFFLOADED_META_KEY,
    _offloadable_text,
    _offloaded_message,
    _result_store_key,
)
from haystack.hooks.tool_result_offloading.types import ToolResultStore
from haystack.token_counters import TokenCounter
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)


@_experimental
class ToolResultOffloadCompactor(Compactor):
    """
    Offload older tool results to a store while keeping compact references in the conversation.

    Unlike `ToolResultOffloadHook`, which runs immediately after a tool and therefore offloads fresh output, this
    compactor runs only when `CompactionHook` detects context pressure. Recent results stay directly available to the
    model until they become old enough to offload. The full text remains available through the reference left in the
    tool-result message.

    Results are considered oldest first. The latest `min_keep_steps` tool-calling Agent steps are always left intact,
    including all results from parallel tool calls in those steps. Only successful, text results are offloaded, using
    the same store abstraction, pointer format, and metadata marker as `ToolResultOffloadHook`.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, ToolResultOffloadCompactor
    from haystack.hooks.tool_result_offloading import FileSystemToolResultStore

    hook = CompactionHook(
        compactor=ToolResultOffloadCompactor(
            store=FileSystemToolResultStore(root="tool_results"),
            min_keep_steps=1,
        ),
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
        self, store: ToolResultStore, *, min_keep_steps: int = 1, min_tokens: int = 200, preview_chars: int = 200
    ) -> None:
        """
        Initialize the compactor with its store and eligibility rules.

        :param store: Where offloaded results are written.
        :param min_keep_steps: The minimum number of recent tool-calling Agent steps whose results remain untouched.
            Must be at least 1, which ensures the current result batch stays in context until the model has acted on
            it.
        :param min_tokens: Only offload tool-result messages that use more than this many tokens. Small results cost
            little and are often worth keeping directly in context.
        :param preview_chars: Number of leading characters of the original result to include in the stored-result
            pointer.
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
        Replace eligible older tool results with references to their stored content.

        Results are considered oldest first and offloading stops as soon as the conversation reaches `target_tokens`.
        This keeps as much recent output directly in context as possible. Results from the most recent
        `min_keep_steps` tool-calling Agent steps are never considered, even when the target cannot otherwise be
        reached.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation and each replacement.
        :returns: The conversation with older tool results offloaded, or None when no result could reduce its size.
        """
        current_tokens = token_counter.count(messages=messages)
        if current_tokens <= target_tokens:
            return None

        result_steps = [
            list(range(start + 1, end))
            for start, end in _agent_step_spans(messages=messages, start=0)
            if end > start + 1
        ]
        protected_positions = {position for step in result_steps[-self.min_keep_steps :] for position in step}

        compacted = list(messages)
        changed = False
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
        """Run store I/O in a worker thread so asynchronous Agent runs do not block the event loop."""
        return await asyncio.to_thread(self.compact, messages, target_tokens, token_counter)

    def _offload(self, message: ChatMessage, index: int, token_counter: TokenCounter) -> tuple[ChatMessage, int] | None:
        """
        Offload one eligible tool-result message and report how many tokens its pointer saves.

        :param message: The tool-result message to consider.
        :param index: The message's position in the conversation, used to make its store key unique within a run.
        :param token_counter: The `TokenCounter` used to measure the original and replacement messages.
        :returns: The offloaded message and tokens saved, or None when the message should stay in context.
        """
        result = message.tool_call_result
        if (
            result is None
            or result.error
            or _OFFLOADED_META_KEY in message.meta
            or _COMPACTION_META_KEY in message.meta
        ):
            return None

        original_tokens = token_counter.count(messages=[message])
        if original_tokens <= self.min_tokens:
            return None

        text = _offloadable_text(result.result)
        if text is None:
            logger.warning(
                "Tool '{tool}' produced a non-text result; leaving it in context. Result offloading currently "
                "supports text results only.",
                tool=result.origin.tool_name,
            )
            return None

        # Include a content digest so different runs sharing a store cannot overwrite one another when the tool-call
        # id and conversation position happen to match. Identical results deliberately reuse the same key.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        tool_call_id = f"{result.origin.id}_{digest}" if result.origin.id else digest
        key = _result_store_key(result.origin.tool_name, tool_call_id, step=index, index=index)
        marked = ChatMessage.from_tool(
            tool_result=result.result,
            origin=result.origin,
            error=result.error,
            meta={
                **message.meta,
                _COMPACTION_META_KEY: {"strategy": "tool_result_offloading", "original_tokens": original_tokens},
            },
        )
        offloaded = _offloaded_message(marked, store=self.store, key=key, text=text, preview_chars=self.preview_chars)
        saved_tokens = original_tokens - token_counter.count(messages=[offloaded])
        # A long store reference or preview can outweigh the original result. In that case, keep the original message
        # so the compactor never grows the context. The store may retain the harmless unreferenced write because the
        # ToolResultStore protocol intentionally does not require deletion support.
        return (offloaded, saved_tokens) if saved_tokens > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor, including its result store and eligibility settings."""
        return default_to_dict(
            self,
            store=self.store.to_dict(),
            min_keep_steps=self.min_keep_steps,
            min_tokens=self.min_tokens,
            preview_chars=self.preview_chars,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResultOffloadCompactor":
        """Deserialize the compactor and reconstruct its result store."""
        init_params = data.get("init_parameters", {})
        if init_params.get("store") is not None:
            deserialize_component_inplace(init_params, key="store")
        return default_from_dict(cls, data)
