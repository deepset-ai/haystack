# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


@component
class FallbackChatGenerator:
    """
    A chat generator wrapper that tries multiple chat generators sequentially.

    It forwards all parameters transparently to the underlying chat generators and returns the first successful result.
    Calls chat generators sequentially until one succeeds. Falls back on any exception raised by a generator.
    If all chat generators fail, it raises a RuntimeError with details.

    Timeout enforcement is fully delegated to the underlying chat generators. The fallback mechanism will only
    work correctly if the underlying chat generators implement proper timeout handling and raise exceptions
    when timeouts occur. For predictable latency guarantees, ensure your chat generators:
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

    def __init__(self, chat_generators: list[ChatGenerator]):
        """
        Creates an instance of FallbackChatGenerator.

        :param chat_generators: A non-empty list of chat generator components to try in order.
        """
        if not chat_generators:
            msg = "'chat_generators' must be a non-empty list"
            raise ValueError(msg)

        self.chat_generators = list(chat_generators)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component, including nested chat generators when they support serialization."""
        return default_to_dict(
            self, chat_generators=[gen.to_dict() for gen in self.chat_generators if hasattr(gen, "to_dict")]
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FallbackChatGenerator:
        """Rebuild the component from a serialized representation, restoring nested chat generators."""
        # Reconstruct nested chat generators from their serialized dicts
        init_params = data.get("init_parameters", {})
        serialized = init_params.get("chat_generators") or []
        deserialized: list[Any] = []
        for g in serialized:
            # Use the generic component deserializer available in Haystack
            holder = {"component": g}
            deserialize_component_inplace(holder, key="component")
            deserialized.append(holder["component"])
        init_params["chat_generators"] = deserialized
        data["init_parameters"] = init_params
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Warm up all underlying chat generators.

        This method calls warm_up() on each underlying generator that supports it.
        """
        for gen in self.chat_generators:
            if hasattr(gen, "warm_up") and callable(gen.warm_up):
                gen.warm_up()

    def _run_single_sync(  # pylint: disable=too-many-positional-arguments
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: Union[dict[str, Any], None],
        tools: Optional[ToolsType],
        streaming_callback: Union[StreamingCallbackT, None],
    ) -> dict[str, Any]:
        return gen.run(
            messages=messages, generation_kwargs=generation_kwargs, tools=tools, streaming_callback=streaming_callback
        )

    async def _run_single_async(  # pylint: disable=too-many-positional-arguments
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: Union[dict[str, Any], None],
        tools: Optional[ToolsType],
        streaming_callback: Union[StreamingCallbackT, None],
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

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Union[dict[str, Any], None] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Union[StreamingCallbackT, None] = None,
    ) -> dict[str, Any]:
        """
        Execute chat generators sequentially until one succeeds.

        :param messages: The conversation history as a list of ChatMessage instances.
        :param generation_kwargs: Optional parameters for the chat generator (e.g., temperature, max_tokens).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for function calling capabilities.
        :param streaming_callback: Optional callable for handling streaming responses.
        :returns: A dictionary with:
            - "replies": Generated ChatMessage instances from the first successful generator.
            - "meta": Execution metadata including successful_chat_generator_index, successful_chat_generator_class,
              total_attempts, failed_chat_generators, plus any metadata from the successful generator.
        :raises RuntimeError: If all chat generators fail.
        """
        failed: list[str] = []
        last_error: Union[BaseException, None] = None

        for idx, gen in enumerate(self.chat_generators):
            gen_name = gen.__class__.__name__
            try:
                result = self._run_single_sync(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_chat_generator_index": idx,
                        "successful_chat_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_chat_generators": failed,
                    }
                )
                return {"replies": replies, "meta": meta}
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                logger.warning(
                    "ChatGenerator {chat_generator} failed with error: {error}", chat_generator=gen_name, error=e
                )
                failed.append(gen_name)
                last_error = e

        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.chat_generators)} chat generators failed. "
            f"Last error: {last_error}. Failed chat generators: [{failed_names}]"
        )
        raise RuntimeError(msg)

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: Union[dict[str, Any], None] = None,
        tools: Optional[ToolsType] = None,
        streaming_callback: Union[StreamingCallbackT, None] = None,
    ) -> dict[str, Any]:
        """
        Asynchronously execute chat generators sequentially until one succeeds.

        :param messages: The conversation history as a list of ChatMessage instances.
        :param generation_kwargs: Optional parameters for the chat generator (e.g., temperature, max_tokens).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for function calling capabilities.
        :param streaming_callback: Optional callable for handling streaming responses.
        :returns: A dictionary with:
            - "replies": Generated ChatMessage instances from the first successful generator.
            - "meta": Execution metadata including successful_chat_generator_index, successful_chat_generator_class,
              total_attempts, failed_chat_generators, plus any metadata from the successful generator.
        :raises RuntimeError: If all chat generators fail.
        """
        failed: list[str] = []
        last_error: Union[BaseException, None] = None

        for idx, gen in enumerate(self.chat_generators):
            gen_name = gen.__class__.__name__
            try:
                result = await self._run_single_async(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_chat_generator_index": idx,
                        "successful_chat_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_chat_generators": failed,
                    }
                )
                return {"replies": replies, "meta": meta}
            except Exception as e:  # noqa: BLE001 - fallback logic should handle any exception
                logger.warning(
                    "ChatGenerator {chat_generator} failed with error: {error}", chat_generator=gen_name, error=e
                )
                failed.append(gen_name)
                last_error = e

        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.chat_generators)} chat generators failed. "
            f"Last error: {last_error}. Failed chat generators: [{failed_names}]"
        )
        raise RuntimeError(msg)
