# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from haystack import logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _agent_step_spans,
    _latest_user_index,
    _leading_system_end,
)
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

    In typical Agent use, `CompactionHook` supplies the target to `compact` after the conversation reaches its
    configured context threshold. Tool results therefore stay directly available to the model until compaction is
    needed.

    The conversation is read as two regions. History runs from the end of the leading system messages up to the latest
    real user message; the current task runs from that user message to the end. Historical tool results are considered
    first, oldest to newest. Current-task results follow in the same order, but the `min_keep_steps` newest
    tool-calling Agent steps remain intact. All results from parallel tool calls belong to the same step and are
    protected together.

    Only successful text-result messages using more than `min_tokens` are eligible. Results already rewritten by
    offloading or another compactor are skipped. Each eligible result is written to the configured `ToolResultStore`
    and replaced with a reference, its original character count, and up to `preview_chars` leading characters. The
    originating tool call, error flag, and existing metadata are preserved.

    Offloading stops once the conversation reaches `target_tokens`. If protected or ineligible results prevent that,
    the compactor returns whatever reduction it can make. It returns None when the conversation already fits or no
    eligible replacement would reduce its size.

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
        :param min_keep_steps: The minimum number of recent tool-calling Agent steps whose results remain untouched,
            even when they exceed the target. Must be at least 1, which ensures the current result batch remains intact
            until the model has acted on it.
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
        Return a conversation with eligible tool results offloaded, or None when no useful reduction is possible.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation and each replacement.
        :returns: The conversation with older tool results offloaded, or None when no result could reduce its size.
        """
        # Skip compaction when the conversation already fits.
        current_tokens = token_counter.count(messages=messages)
        if current_tokens <= target_tokens:
            return None

        compacted = list(messages)
        changed = False
        # Iterate over eligible tool-result messages in offload priority order, replacing them until the target is met.
        for index in self._candidate_positions(messages=messages):
            message = messages[index]
            replacement = self._offload(message=message, index=index, token_counter=token_counter)
            # Skip ineligible results and those that would not reduce context.
            if replacement is None:
                continue
            replacement_msg, saved_tokens = replacement
            compacted[index] = replacement_msg
            current_tokens -= saved_tokens
            changed = True
            if current_tokens <= target_tokens:
                break

        return compacted if changed else None

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Asynchronous version of `compact`. Offloading is performed in a thread to avoid blocking the event loop.

        :param messages: The conversation to compact, oldest to newest.
        :param target_tokens: The size the compacted conversation should come in under.
        :param token_counter: The `TokenCounter` used to measure the conversation and each replacement.
        :returns: The conversation with older tool results offloaded, or None when no result could reduce its size.
        """
        return await asyncio.to_thread(
            self.compact, messages=messages, target_tokens=target_tokens, token_counter=token_counter
        )

    def _candidate_positions(self, messages: list[ChatMessage]) -> list[int]:
        """Return tool-result positions in offload priority order."""
        task_index = _latest_user_index(messages=messages)
        current_task_start = task_index + 1 if task_index is not None else _leading_system_end(messages=messages)
        historical = [index for index in range(current_task_start) if messages[index].tool_call_result is not None]

        current_steps = [
            list(range(start + 1, end))
            for start, end in _agent_step_spans(messages=messages, start=current_task_start)
            if end > start + 1
        ]
        current = [index for step in current_steps[: -self.min_keep_steps] for index in step]
        return historical + current

    def _offload(self, message: ChatMessage, index: int, token_counter: TokenCounter) -> tuple[ChatMessage, int] | None:
        """
        Offload one eligible tool-result message and report how many tokens its pointer saves.

        :param message: The tool-result message to consider.
        :param index: The message's position in the conversation, used to make its store key unique within a run.
        :param token_counter: The `TokenCounter` used to measure the original and replacement messages.
        :returns: The offloaded message and tokens saved, or None when the message should stay in context.
        """
        result = message.tool_call_result
        # Only offload successful text results that are not already rewritten.
        if (
            result is None
            or result.error
            or _OFFLOADED_META_KEY in message.meta
            or _COMPACTION_META_KEY in message.meta
        ):
            return None

        # Count the original message's tokens and skip it if it's too small to offload.
        original_tokens = token_counter.count(messages=[message])
        if original_tokens <= self.min_tokens:
            return None

        # Only offload results that can be represented as text. Non-text results are left in context.
        text = _offloadable_text(content=result.result)
        if text is None:
            logger.warning(
                "Tool '{tool}' produced a non-text result; leaving it in context. Result offloading currently "
                "supports text results only.",
                tool=result.origin.tool_name,
            )
            return None

        key = _result_store_key(
            tool_name=result.origin.tool_name, tool_call_id=result.origin.id, step=index, index=index
        )
        offloaded = _offloaded_message(
            message=message,
            store=self.store,
            key=key,
            text=text,
            preview_chars=self.preview_chars,
            additional_meta={
                _COMPACTION_META_KEY: {"strategy": "tool_result_offloading", "original_tokens": original_tokens}
            },
        )
        saved_tokens = original_tokens - token_counter.count(messages=[offloaded])
        # Keep the original when its pointer would not reduce context.
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
        return default_from_dict(cls=cls, data=data)
