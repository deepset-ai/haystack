# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCall,
    select_streaming_callback,
)
from haystack.tools.tool import Tool, _check_duplicate_tool_names, deserialize_tools_inplace
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


@component
class OpenAIChatGenerator:
    """
    Completes chats using OpenAI's large language models (LLMs).

    It works with the gpt-4 and gpt-3.5-turbo models and supports streaming responses
    from OpenAI API. It uses [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage)
    format in input and output.

    You can customize how the text is generated by passing parameters to the
    OpenAI API. Use the `**generation_kwargs` argument when you initialize
    the component or when you run it. Any parameter that works with
    `openai.ChatCompletion.create` will work here too.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat).

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenAIChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    Output:
    ```
    {'replies':
        [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
            that focuses on enabling computers to understand, interpret, and generate human language in
            a way that is meaningful and useful.',
         role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
         meta={'model': 'gpt-4o-mini', 'index': 0, 'finish_reason': 'stop',
         'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})
        ]
    }
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: bool = False,
    ):
        """
        Creates an instance of OpenAIChatGenerator. Unless specified otherwise in `model`, uses OpenAI's gpt-4o-mini

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key: The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument.
        :param api_base_url: An optional base URL.
        :param organization: Your organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param generation_kwargs: Other parameters to use for the model. These parameters are sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. For example, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param tools:
            A list of tools for which the model can prepare calls.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.tools = tools
        self.tools_strict = tools_strict

        _check_duplicate_tool_names(tools)

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        client_args: Dict[str, Any] = {
            "api_key": api_key.resolve_value(),
            "organization": organization,
            "base_url": api_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        self.client = OpenAI(**client_args)
        self.async_client = AsyncOpenAI(**client_args)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=[tool.to_dict() for tool in self.tools] if self.tools else None,
            tools_strict=self.tools_strict,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = streaming_callback or self.streaming_callback

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        chat_completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            **api_args
        )

        is_streaming = isinstance(chat_completion, Stream)
        assert is_streaming or streaming_callback is None

        if is_streaming:
            completions = self._handle_stream_response(
                chat_completion,  # type: ignore
                streaming_callback,  # type: ignore
            )
        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                self._convert_chat_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message.meta)

        return {"replies": completions}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Asynchronously invokes chat completion based on the provided messages and generation parameters.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Must be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(self.streaming_callback, streaming_callback, requires_async=True)  # type: ignore

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )

        chat_completion: Union[
            AsyncStream[ChatCompletionChunk], ChatCompletion
        ] = await self.async_client.chat.completions.create(**api_args)

        is_streaming = isinstance(chat_completion, AsyncStream)
        assert is_streaming or streaming_callback is None

        if is_streaming:
            completions = await self._handle_async_stream_response(
                chat_completion,  # type: ignore
                streaming_callback,  # type: ignore
            )
        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                self._convert_chat_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message.meta)

        return {"replies": completions}

    def _prepare_api_call(  # noqa: PLR0913
        self,
        *,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

        tools = tools or self.tools
        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        _check_duplicate_tool_names(tools)

        openai_tools = {}
        if tools:
            tool_definitions = []
            for t in tools:
                function_spec = {**t.tool_spec}
                if tools_strict:
                    function_spec["strict"] = True
                    function_spec["parameters"]["additionalProperties"] = False
                tool_definitions.append({"type": "function", "function": function_spec})
            openai_tools = {"tools": tool_definitions}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)
        if is_streaming and num_responses > 1:
            raise ValueError("Cannot stream multiple responses, please set n=1.")

        return {
            "model": self.model,
            "messages": openai_formatted_messages,  # type: ignore[arg-type] # openai expects list of specific message types
            "stream": streaming_callback is not None,
            "n": num_responses,
            **openai_tools,
            **generation_kwargs,
        }

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> List[ChatMessage]:
        chunks: List[StreamingChunk] = []
        chunk = None

        for chunk in chat_completion:  # pylint: disable=not-an-iterable
            assert len(chunk.choices) == 1, "Streaming responses should have only one choice."
            chunk_delta: StreamingChunk = self._convert_chat_completion_chunk_to_streaming_chunk(chunk)
            chunks.append(chunk_delta)

            callback(chunk_delta)

        return [self._convert_streaming_chunks_to_chat_message(chunk, chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream, callback: AsyncStreamingCallbackT
    ) -> List[ChatMessage]:
        chunks: List[StreamingChunk] = []
        chunk = None

        async for chunk in chat_completion:  # pylint: disable=not-an-iterable
            # choices is an empty array for usage_chunk when include_usage is set to True
            if chunk.usage is not None:
                chunk_delta = self._convert_usage_chunk_to_streaming_chunk(chunk)

            else:
                assert len(chunk.choices) == 1, "Streaming responses should have only one choice."
                chunk_delta: StreamingChunk = self._convert_chat_completion_chunk_to_streaming_chunk(chunk)
            chunks.append(chunk_delta)

            await callback(chunk_delta)
        return [self._convert_streaming_chunks_to_chat_message(chunk, chunks)]

    def _check_finish_reason(self, meta: Dict[str, Any]) -> None:
        if meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=meta["index"],
                finish_reason=meta["finish_reason"],
            )
        if meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index {index} has been truncated due to the content filter.",
                index=meta["index"],
                finish_reason=meta["finish_reason"],
            )

    def _convert_streaming_chunks_to_chat_message(
        self, chunk: ChatCompletionChunk, chunks: List[StreamingChunk]
    ) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.

        :param chunk: The last chunk returned by the OpenAI API.
        :param chunks: The list of all `StreamingChunk` objects.
        """
        text = "".join([chunk.content for chunk in chunks])
        tool_calls = []

        # Process tool calls if present in any chunk
        tool_call_data: Dict[str, Dict[str, str]] = {}  # Track tool calls by index
        for chunk_payload in chunks:
            tool_calls_meta = chunk_payload.meta.get("tool_calls")
            if tool_calls_meta is not None:
                for delta in tool_calls_meta:
                    # We use the index of the tool call to track it across chunks since the ID is not always provided
                    if delta.index not in tool_call_data:
                        tool_call_data[delta.index] = {"id": "", "name": "", "arguments": ""}

                    # Save the ID if present
                    if delta.id is not None:
                        tool_call_data[delta.index]["id"] = delta.id

                    if delta.function is not None:
                        if delta.function.name is not None:
                            tool_call_data[delta.index]["name"] += delta.function.name
                        if delta.function.arguments is not None:
                            tool_call_data[delta.index]["arguments"] += delta.function.arguments

        # Convert accumulated tool call data into ToolCall objects
        for call_data in tool_call_data.values():
            try:
                arguments = json.loads(call_data["arguments"])
                tool_calls.append(ToolCall(id=call_data["id"], tool_name=call_data["name"], arguments=arguments))
            except json.JSONDecodeError:
                logger.warning(
                    "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                    "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                    "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                    _id=call_data["id"],
                    _name=call_data["name"],
                    _arguments=call_data["arguments"],
                )

        finish_reason = (chunks[-2] if chunk.usage else chunks[-1]).meta.get("finish_reason")

        meta = {
            "model": chunk.model,
            "index": 0,
            "finish_reason": finish_reason,
            "completion_start_time": chunks[0].meta.get("received_at"),  # first chunk received
            "usage": chunk.usage or {},
        }

        return ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, meta=meta)

    def _convert_chat_completion_to_chat_message(self, completion: ChatCompletion, choice: Choice) -> ChatMessage:
        """
        Converts the non-streaming response from the OpenAI API to a ChatMessage.

        :param completion: The completion returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The ChatMessage.
        """
        message: ChatCompletionMessage = choice.message
        text = message.content
        tool_calls = []
        if openai_tool_calls := message.tool_calls:
            for openai_tc in openai_tool_calls:
                arguments_str = openai_tc.function.arguments
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(ToolCall(id=openai_tc.id, tool_name=openai_tc.function.name, arguments=arguments))
                except json.JSONDecodeError:
                    logger.warning(
                        "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                        "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                        _id=openai_tc.id,
                        _name=openai_tc.function.name,
                        _arguments=arguments_str,
                    )

        chat_message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls)
        chat_message._meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage or {}),
            }
        )
        return chat_message

    def _convert_chat_completion_chunk_to_streaming_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk:
        """
        Converts the streaming response chunk from the OpenAI API to a StreamingChunk.

        :param chunk: The chunk returned by the OpenAI API.

        :returns:
            The StreamingChunk.
        """
        # we stream the content of the chunk if it's not a tool or function call
        choice: ChunkChoice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        # but save the tool calls and function call in the meta if they are present
        # and then connect the chunks in the _convert_streaming_chunks_to_chat_message method
        chunk_message.meta.update(
            {
                "model": chunk.model,
                "index": choice.index,
                "tool_calls": choice.delta.tool_calls,
                "finish_reason": choice.finish_reason,
                "received_at": datetime.now().isoformat(),
            }
        )
        return chunk_message

    def _convert_usage_chunk_to_streaming_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk:
        """
        Converts the usage chunk from the OpenAI API to a StreamingChunk.

        :param chunk: The usage chunk returned by the OpenAI API.

        :returns:
            The StreamingChunk.
        """
        chunk_message = StreamingChunk(content="")
        chunk_message.meta.update(
            {"model": chunk.model, "usage": chunk.usage, "received_at": datetime.now().isoformat()}
        )
        return chunk_message
