# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _normalize_messages
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FinishReason,
    StreamingCallbackT,
    StreamingChunk,
    select_streaming_callback,
)
from haystack.dataclasses.streaming_chunk import ToolCallDelta, _invoke_streaming_callback
from haystack.tools import ToolsType
from haystack.utils import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)

# A callable that derives a response from the input messages. It receives the (normalized) list of input
# `ChatMessage` objects and returns either the text of the assistant reply or a full `ChatMessage`.
ResponseFn = Callable[[list[ChatMessage]], str | ChatMessage]


@component
class MockChatGenerator:
    """
    A Chat Generator that returns predefined responses without calling any API.

    It is a drop-in replacement for real Chat Generators (such as `OpenAIChatGenerator`) in tests, smoke tests, and
    quick prototypes. It implements the same interface (`run`, `run_async`, streaming, serialization) but never
    contacts an external service, so it is fully deterministic and free to run.

    The response is selected based on how the component is configured:

    - **Fixed response**: pass a single string or `ChatMessage`. The same reply is returned on every call.
      Any `ChatMessage` passed as a response must have the `assistant` role.
    - **Cycling responses**: pass a list of strings and/or `ChatMessage` objects. Each call returns the next item,
      wrapping around to the start once the list is exhausted. This is useful to drive multi-step flows such as
      Agents, where the first call returns a tool call and a later call returns the final answer.
    - **Dynamic response**: pass a `response_fn` callable that receives the input messages and returns the reply.
      This is useful when the reply should depend on the input, for example to echo back part of the prompt.
    - **Echo (default)**: with no configuration, the component echoes back the text of the last message that has
      text content. This makes it usable out of the box for quick prototyping.

    Pass `ChatMessage` objects (rather than plain strings) to return tool calls or reasoning content, which is handy
    for exercising tool-calling pipelines without a real model.

    ### Usage example

    ```python
    from haystack.components.generators.chat import MockChatGenerator
    from haystack.dataclasses import ChatMessage, ToolCall

    # Fixed response
    generator = MockChatGenerator(responses="Hello, this is a mock response.")
    result = generator.run([ChatMessage.from_user("Hi!")])
    print(result["replies"][0].text)  # "Hello, this is a mock response."

    # Cycling responses to drive an Agent-like loop
    generator = MockChatGenerator(
        responses=[
            ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="search", arguments={"query": "Haystack"})]),
            "Here is the final answer.",
        ]
    )
    ```
    """

    def __init__(
        self,
        responses: str | ChatMessage | Sequence[str | ChatMessage] | None = None,
        *,
        response_fn: ResponseFn | None = None,
        model: str = "mock-model",
        meta: dict[str, Any] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> None:
        """
        Creates an instance of MockChatGenerator.

        :param responses: The predefined response(s) to return. Accepts a single string or `ChatMessage` (returned on
            every call), or a non-empty list of strings and/or `ChatMessage` objects that are returned in order,
            cycling back to the start once exhausted. Strings are wrapped into assistant `ChatMessage` objects, and any
            `ChatMessage` passed must have the `assistant` role. Mutually exclusive with `response_fn`. If neither is
            provided, the component echoes the last message with text content.
        :param response_fn: An optional callable that receives the input messages and returns the reply as a string or
            an assistant `ChatMessage`. Use this for input-dependent responses. Mutually exclusive with `responses`. To
            support serialization, pass a named function (lambdas and nested functions cannot be serialized).
        :param model: The model name reported in the response metadata. Purely cosmetic; no model is loaded.
        :param meta: Additional metadata merged into the `meta` of every returned `ChatMessage`. A per-response
            `ChatMessage`'s own metadata takes precedence over this value.
        :param streaming_callback: An optional callback invoked with `StreamingChunk` objects reconstructed from the
            predefined response. It lets the mock exercise streaming code paths without a real model.
        :raises ValueError: If both `responses` and `response_fn` are provided, if `responses` is an empty list, or if
            a `ChatMessage` response does not have the `assistant` role.
        """
        if responses is not None and response_fn is not None:
            raise ValueError("Pass either 'responses' or 'response_fn', not both.")

        self._responses = self._normalize_responses(responses)
        self.response_fn = response_fn
        self.model = model
        self.meta = meta or {}
        self.streaming_callback = streaming_callback
        self._call_count = 0
        self._is_warmed_up = False

    @staticmethod
    def _normalize_responses(
        responses: str | ChatMessage | Sequence[str | ChatMessage] | None,
    ) -> list[ChatMessage] | None:
        """Normalize the `responses` argument into a non-empty list of `ChatMessage`, or `None` for echo mode."""
        if responses is None:
            return None

        items: list[str | ChatMessage]
        if isinstance(responses, (str, ChatMessage)):
            items = [responses]
        elif isinstance(responses, Sequence):
            items = list(responses)
        else:
            raise TypeError(f"'responses' must be a string, ChatMessage, or a sequence of them, got {type(responses)}.")

        if len(items) == 0:
            raise ValueError("'responses' must not be an empty list.")

        normalized: list[ChatMessage] = []
        for item in items:
            if isinstance(item, str):
                normalized.append(ChatMessage.from_assistant(item))
            elif isinstance(item, ChatMessage):
                if item.role != ChatRole.ASSISTANT:
                    raise ValueError(
                        f"Each ChatMessage response must have the 'assistant' role, got '{item.role.value}'."
                    )
                normalized.append(item)
            else:
                raise TypeError(f"Each response must be a string or ChatMessage, got {type(item)}.")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        responses = [msg.to_dict() for msg in self._responses] if self._responses is not None else None
        response_fn = serialize_callable(self.response_fn) if self.response_fn is not None else None
        streaming_callback = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            responses=responses,
            response_fn=response_fn,
            model=self.model,
            meta=self.meta,
            streaming_callback=streaming_callback,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockChatGenerator:
        """Deserialize the component from a dictionary."""
        init_params = data.get("init_parameters", {})
        responses = init_params.get("responses")
        if responses is not None:
            init_params["responses"] = [ChatMessage.from_dict(msg) for msg in responses]
        response_fn = init_params.get("response_fn")
        if response_fn:
            init_params["response_fn"] = deserialize_callable(response_fn)
        streaming_callback = init_params.get("streaming_callback")
        if streaming_callback:
            init_params["streaming_callback"] = deserialize_callable(streaming_callback)
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """No-op warm up, provided for interface compatibility with real Chat Generators."""
        self._is_warmed_up = True

    @staticmethod
    def _echo_text(messages: list[ChatMessage]) -> str | None:
        """Return the text of the last message that has text content, for echo mode."""
        for message in reversed(messages):
            if message.text:
                return message.text
        return None

    @staticmethod
    def _coerce_to_message(result: str | ChatMessage) -> ChatMessage:
        """Turn the output of `response_fn` into a `ChatMessage`, wrapping strings and requiring the assistant role."""
        if isinstance(result, str):
            return ChatMessage.from_assistant(result)
        if isinstance(result, ChatMessage):
            if result.role != ChatRole.ASSISTANT:
                raise ValueError(f"'response_fn' must return an assistant ChatMessage, got '{result.role.value}'.")
            return result
        raise TypeError(f"'response_fn' must return a string or ChatMessage, got {type(result)}.")

    @staticmethod
    def _estimate_usage(messages: list[ChatMessage], reply: ChatMessage) -> dict[str, int]:
        """
        Roughly estimate token usage as whitespace-separated word counts.

        This is an approximation (not real tokenization) intended to give downstream code realistic-looking metadata.
        """
        prompt_tokens = sum(len((message.text or "").split()) for message in messages)
        completion_tokens = len((reply.text or "").split())
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _build_meta(self, messages: list[ChatMessage], base: ChatMessage) -> dict[str, Any]:
        """Build the metadata attached to the returned reply, merging defaults, init meta, and per-response meta."""
        meta: dict[str, Any] = {
            "model": self.model,
            "index": 0,
            "finish_reason": "tool_calls" if base.tool_calls else "stop",
            "usage": self._estimate_usage(messages, base),
        }
        meta.update(self.meta)
        meta.update(base.meta)
        return meta

    def _build_reply(self, messages: list[ChatMessage]) -> ChatMessage | None:
        """Select and finalize the reply for the given input messages. Returns `None` when there is nothing to echo."""
        if self.response_fn is not None:
            base = self._coerce_to_message(self.response_fn(messages))
        elif self._responses is not None:
            base = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
        else:
            text = self._echo_text(messages)
            if text is None:
                return None
            base = ChatMessage.from_assistant(text)

        return replace(base, _meta=self._build_meta(messages, base))

    def _make_chunks(self, reply: ChatMessage) -> list[StreamingChunk]:
        """Reconstruct streaming chunks from a finalized reply so streaming callbacks can be exercised."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []

        # Stream the text content word by word in content block 0.
        parts = re.findall(r"\S+\s*", reply.text) if reply.text else []
        for idx, part in enumerate(parts):
            chunks.append(
                StreamingChunk(
                    content=part, component_info=component_info, index=0, start=(idx == 0), meta={"model": self.model}
                )
            )

        # Stream each tool call in its own content block.
        block_index = 1 if parts else 0
        for tool_call in reply.tool_calls:
            chunks.append(
                StreamingChunk(
                    content="",
                    component_info=component_info,
                    index=block_index,
                    start=True,
                    tool_calls=[
                        ToolCallDelta(
                            index=block_index,
                            tool_name=tool_call.tool_name,
                            arguments=json.dumps(tool_call.arguments),
                            id=tool_call.id,
                        )
                    ],
                    meta={"model": self.model},
                )
            )
            block_index += 1

        if not chunks:
            chunks.append(
                StreamingChunk(content="", component_info=component_info, index=0, meta={"model": self.model})
            )

        finish_reason: FinishReason = "tool_calls" if reply.tool_calls else "stop"
        last = chunks[-1]
        chunks[-1] = replace(last, finish_reason=finish_reason, meta={**last.meta, "finish_reason": finish_reason})
        return chunks

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
        *,
        tools: ToolsType | None = None,  # noqa: ARG002
        tools_strict: bool | None = None,  # noqa: ARG002
    ) -> dict[str, list[ChatMessage]]:
        """
        Return a predefined reply for the given messages without calling any API.

        The signature mirrors `OpenAIChatGenerator.run` so the mock can be used as a positional drop-in replacement.

        :param messages: The conversation history as a list of `ChatMessage` instances or a single string.
        :param streaming_callback: An optional callback invoked with reconstructed `StreamingChunk` objects. Overrides
            the callback set at initialization.
        :param generation_kwargs: Accepted for interface compatibility and ignored.
        :param tools: Accepted for interface compatibility and ignored.
        :param tools_strict: Accepted for interface compatibility and ignored.
        :returns: A dictionary with a single key `replies` containing the predefined reply as a list of one
            `ChatMessage` (empty in echo mode when there is no message to echo).
        """
        self.warm_up()

        messages = _normalize_messages(messages)
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        reply = self._build_reply(messages)
        if reply is None:
            return {"replies": []}

        if streaming_callback is not None:
            for chunk in self._make_chunks(reply):
                streaming_callback(chunk)

        return {"replies": [reply]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
        *,
        tools: ToolsType | None = None,  # noqa: ARG002
        tools_strict: bool | None = None,  # noqa: ARG002
    ) -> dict[str, list[ChatMessage]]:
        """
        Asynchronously return a predefined reply for the given messages without calling any API.

        The signature mirrors `OpenAIChatGenerator.run_async` so the mock can be used as a positional drop-in
        replacement.

        :param messages: The conversation history as a list of `ChatMessage` instances or a single string.
        :param streaming_callback: An optional callback invoked with reconstructed `StreamingChunk` objects. Overrides
            the callback set at initialization.
        :param generation_kwargs: Accepted for interface compatibility and ignored.
        :param tools: Accepted for interface compatibility and ignored.
        :param tools_strict: Accepted for interface compatibility and ignored.
        :returns: A dictionary with a single key `replies` containing the predefined reply as a list of one
            `ChatMessage` (empty in echo mode when there is no message to echo).
        """
        if not self._is_warmed_up:
            self.warm_up()

        messages = _normalize_messages(messages)
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        reply = self._build_reply(messages)
        if reply is None:
            return {"replies": []}

        if streaming_callback is not None:
            for chunk in self._make_chunks(reply):
                await _invoke_streaming_callback(streaming_callback, chunk)

        return {"replies": [reply]}
