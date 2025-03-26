# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, patch

from openai import AsyncOpenAI, OpenAIError
import pytest
from datetime import datetime
import os

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat import chat_completion_chunk

from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.components.generators.chat.openai import OpenAIChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_chat_completion_chunk_with_tools(openai_mock_stream_async):
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            chat_completion_chunk.ChoiceDeltaToolCall(
                                index=0,
                                id="123",
                                type="function",
                                function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                    name="weather", arguments='{"city": "Paris"}'
                                ),
                            )
                        ],
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage=None,
        )
        mock_chat_completion_create.return_value = openai_mock_stream_async(completion)
        yield mock_chat_completion_create


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=lambda x: x,
    )
    return [tool]


class TestOpenAIChatGeneratorAsync:
    def test_init_should_also_create_async_client_with_same_args(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            api_base_url="test-base-url",
            organization="test-organization",
            timeout=30,
            max_retries=5,
        )

        assert isinstance(component.async_client, AsyncOpenAI)
        assert component.async_client.api_key == "test-api-key"
        assert component.async_client.organization == "test-organization"
        assert component.async_client.base_url == "test-base-url/"
        assert component.async_client.timeout == 30
        assert component.async_client.max_retries == 5

    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, openai_mock_async_chat_completion):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = await component.run_async(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_with_params_async(self, chat_messages, openai_mock_async_chat_completion):
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = await component.run_async(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = openai_mock_async_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the tools are not passed to the OpenAI API (the generator is initialized without tools)
        assert "tools" not in kwargs

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_with_params_streaming_async(self, chat_messages, openai_mock_async_chat_completion_chunk):
        streaming_callback_called = False

        async def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        response = await component.run_async(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see openai_mock_chat_completion_chunk

    @pytest.mark.asyncio
    async def test_run_with_streaming_callback_in_run_method_async(
        self, chat_messages, openai_mock_async_chat_completion_chunk
    ):
        streaming_callback_called = False

        async def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = await component.run_async(chat_messages, streaming_callback=streaming_callback)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see openai_mock_chat_completion_chunk

    @pytest.mark.asyncio
    async def test_run_with_tools_async(self, tools):
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
        ) as mock_chat_completion_create:
            completion = ChatCompletion(
                id="foo",
                model="gpt-4",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        logprobs=None,
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="123",
                                    type="function",
                                    function=Function(name="weather", arguments='{"city": "Paris"}'),
                                )
                            ],
                        ),
                    )
                ],
                created=int(datetime.now().timestamp()),
                usage=CompletionUsage(
                    completion_tokens=40,
                    prompt_tokens=57,
                    total_tokens=97,
                    completion_tokens_details=CompletionTokensDetails(
                        accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                    ),
                    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
                ),
            )

            mock_chat_completion_create.return_value = completion

            component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"), tools=tools, tools_strict=True)
            response = await component.run_async([ChatMessage.from_user("What's the weather like in Paris?")])

        # ensure that the tools are passed to the OpenAI API
        function_spec = {**tools[0].tool_spec}
        function_spec["strict"] = True
        function_spec["parameters"]["additionalProperties"] = False
        assert mock_chat_completion_create.call_args[1]["tools"] == [{"type": "function", "function": function_spec}]

        assert len(response["replies"]) == 1
        message = response["replies"][0]

        assert not message.texts
        assert not message.text

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert message.meta["usage"]["completion_tokens"] == 40

    @pytest.mark.asyncio
    async def test_run_with_tools_streaming_async(self, mock_chat_completion_chunk_with_tools, tools):
        streaming_callback_called = False

        async def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        response = await component.run_async(chat_messages, tools=tools)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

        message = response["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIChatGenerator(generation_kwargs={"n": 1})
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_wrong_model_async(self, chat_messages):
        component = OpenAIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            await component.run_async(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_streaming_async(self):
        counter = 0
        responses = ""

        async def callback(chunk: StreamingChunk):
            nonlocal counter
            nonlocal responses
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = OpenAIChatGenerator(
            streaming_callback=callback, generation_kwargs={"stream_options": {"include_usage": True}}
        )
        results = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert counter > 1
        assert "Paris" in responses

        # check that the completion_start_time is set and valid ISO format
        assert "completion_start_time" in message.meta
        assert datetime.fromisoformat(message.meta["completion_start_time"]) <= datetime.now()

        assert isinstance(message.meta["usage"], dict)
        assert message.meta["usage"]["prompt_tokens"] > 0
        assert message.meta["usage"]["completion_tokens"] > 0
        assert message.meta["usage"]["total_tokens"] > 0

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_with_tools_async(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenAIChatGenerator(tools=tools)
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
