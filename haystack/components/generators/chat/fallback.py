# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


@component
class FallbackChatGenerator:
    """
    A chat generator wrapper that tries multiple chat generators sequentially.

    It forwards all parameters transparently to the underlying generators and returns the first successful result.
    Calls generators sequentially until one succeeds. Falls back on any exception raised by a generator.
    If all generators fail, it raises a RuntimeError with details.

    Timeout enforcement is fully delegated to the underlying generators. The fallback mechanism will only
    work correctly if the underlying chat generators implement proper timeout handling and raise exceptions
    when timeouts occur. For predictable latency guarantees, ensure your generators:
    - Support a `timeout` parameter in their initialization
    - Implement timeout as total wall-clock time (shared deadline for both streaming and non-streaming)
    - Raise timeout exceptions (e.g., TimeoutError, asyncio.TimeoutError, httpx.TimeoutException) when exceeded

    Note: Most well-implemented chat generators (OpenAI, Anthropic, Cohere, etc.) support timeout parameters
    with consistent semantics. For HTTP-based LLM providers, a single timeout value (e.g., `timeout=30`)
    typically applies to all connection phases: connection setup, read, write, and pool. For streaming
    responses, read timeout is the maximum gap between chunks. For non-streaming, it's the time limit for
    receiving the complete response.

    Failover is automatically triggered when a generator raises any exception, including:
    - Timeout errors (if the generator implements and raises them)
    - Rate limit errors (429)
    - Authentication errors (401)
    - Context length errors (400)
    - Server errors (500+)
    - Any other exception
    """

    def __init__(self, generators: list[ChatGenerator]):
        """
        :param generators: A non-empty list of chat generator components to try in order.
        """
        if not generators:
            msg = "'generators' must be a non-empty list"
            raise ValueError(msg)

        # Validation via duck-typing: require a callable 'run' method
        for gen in generators:
            if not hasattr(gen, "run") or not callable(gen.run):
                msg = "All items in 'generators' must expose a callable 'run' method (duck-typed ChatGenerator)"
                raise TypeError(msg)

        self.generators = list(generators)

    # ---------------------- Serialization ----------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the component, including nested generators when they support serialization."""
        return default_to_dict(self, generators=[gen.to_dict() for gen in self.generators if hasattr(gen, "to_dict")])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FallbackChatGenerator:
        """Rebuild the component from a serialized representation, restoring nested generators."""
        # Reconstruct nested generators from their serialized dicts
        init_params = data.get("init_parameters", {})
        serialized = init_params.get("generators") or []
        deserialized: list[Any] = []
        for g in serialized:
            # Use the generic component deserializer available in Haystack
            holder = {"component": g}
            deserialize_component_inplace(holder, key="component")
            deserialized.append(holder["component"])
        init_params["generators"] = deserialized
        data["init_parameters"] = init_params
        return default_from_dict(cls, data)

    # ---------------------- Execution helpers ----------------------
    def _run_single_sync(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: (list[Tool] | Toolset) | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        return gen.run(
            messages=messages, generation_kwargs=generation_kwargs, tools=tools, streaming_callback=streaming_callback
        )

    async def _run_single_async(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: (list[Tool] | Toolset) | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        if hasattr(gen, "run_async") and callable(gen.run_async):
            return await gen.run_async(
                messages=messages,
                generation_kwargs=generation_kwargs,
                tools=tools,
                streaming_callback=streaming_callback,
            )
        return await asyncio.to_thread(
            gen.run,
            messages=messages,
            generation_kwargs=generation_kwargs,
            tools=tools,
            streaming_callback=streaming_callback,
        )

    # ---------------------- Public API ----------------------
    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        """Execute generators sequentially until one succeeds, returning its replies and enriched metadata."""
        failed: list[str] = []
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.generators):
            gen_name = gen.__class__.__name__
            try:
                result = self._run_single_sync(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_generator_index": idx,
                        "successful_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_generators": failed,
                    }
                )
                return {"replies": replies, "meta": meta}
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                logger.warning("Generator %s failed with error: %s", gen_name, e)
                failed.append(gen_name)
                last_error = e

        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.generators)} generators failed. "
            f"Last error: {last_error}. Failed generators: [{failed_names}]"
        )
        raise RuntimeError(msg)

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        """Asynchronously execute generators in order, returning the first successful result with metadata."""
        failed: list[str] = []
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.generators):
            gen_name = gen.__class__.__name__
            try:
                result = await self._run_single_async(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_generator_index": idx,
                        "successful_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_generators": failed,
                    }
                )
                return {"replies": replies, "meta": meta}
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                logger.warning("Generator %s failed with error: %s", gen_name, e)
                failed.append(gen_name)
                last_error = e

        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.generators)} generators failed. "
            f"Last error: {last_error}. Failed generators: [{failed_names}]"
        )
        raise RuntimeError(msg)
