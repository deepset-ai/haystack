# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch, MagicMock
import pytest


import logging
import os
from datetime import datetime

from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat import chat_completion_chunk

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.tools.toolset import Toolset


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_chat_completion_chunk_with_tools(openai_mock_stream):
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
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
        )
        mock_chat_completion_create.return_value = openai_mock_stream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


def mock_tool_function(x):
    return x


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=mock_tool_function,
    )

    return [tool]


class TestOpenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.timeout == 30
        assert component.client.max_retries == 5
        assert component.tools is None
        assert not component.tools_strict
        assert component.http_client_kwargs is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            OpenAIChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=40.0,
            max_retries=1,
            tools=[tool],
            tools_strict=True,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 40.0
        assert component.client.max_retries == 1
        assert component.tools == [tool]
        assert component.tools_strict
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {},
                "tools": None,
                "tools_strict": False,
                "max_retries": None,
                "timeout": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
            tools_strict=True,
            max_retries=10,
            timeout=100.0,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "max_retries": 10,
                "timeout": 100.0,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "inputs_from_state": None,
                            "name": "name",
                            "outputs_to_state": None,
                            "outputs_to_string": None,
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "max_retries": 10,
                "timeout": 100.0,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }
        component = OpenAIChatGenerator.from_dict(data)

        assert isinstance(component, OpenAIChatGenerator)
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert component.tools_strict
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": None,
            },
        }
        with pytest.raises(ValueError):
            OpenAIChatGenerator.from_dict(data)

    def test_run(self, chat_messages, openai_mock_chat_completion):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, openai_mock_chat_completion):
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = openai_mock_chat_completion.call_args
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

    def test_run_with_params_streaming(self, chat_messages, openai_mock_chat_completion_chunk):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        response = component.run(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see openai_mock_chat_completion_chunk

    def test_run_with_streaming_callback_in_run_method(self, chat_messages, openai_mock_chat_completion_chunk):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages, streaming_callback=streaming_callback)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see openai_mock_chat_completion_chunk

    def test_run_with_wrapped_stream_simulation(self, chat_messages, openai_mock_stream):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True
            assert isinstance(chunk, StreamingChunk)

        chunk = ChatCompletionChunk(
            id="id",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[chat_completion_chunk.Choice(index=0, delta=chat_completion_chunk.ChoiceDelta(content="Hello"))],
            created=int(datetime.now().timestamp()),
        )

        # Here we wrap the OpenAI stream in a MagicMock
        # This is to simulate the behavior of some tools like Weave (https://github.com/wandb/weave)
        # which wrap the OpenAI stream in their own stream
        wrapped_openai_stream = MagicMock()
        wrapped_openai_stream.__iter__.return_value = iter([chunk])

        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))

        with patch.object(
            component.client.chat.completions, "create", return_value=wrapped_openai_stream
        ) as mock_create:
            response = component.run(chat_messages, streaming_callback=streaming_callback)

            mock_create.assert_called_once()
            assert streaming_callback_called
            assert "replies" in response
            assert "Hello" in response["replies"][0].text

    def test_check_abnormal_completions(self, caplog):
        caplog.set_level(logging.INFO)
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        messages = [
            ChatMessage.from_assistant(
                "", meta={"finish_reason": "content_filter" if i % 2 == 0 else "length", "index": i}
            )
            for i, _ in enumerate(range(4))
        ]

        for m in messages:
            component._check_finish_reason(m.meta)

        # check truncation warning
        message_template = (
            "The completion for index {index} has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

        for index in [1, 3]:
            assert caplog.records[index].message == message_template.format(index=index)

        # check content filter warning
        message_template = "The completion for index {index} has been truncated due to the content filter."
        for index in [0, 2]:
            assert caplog.records[index].message == message_template.format(index=index)

    def test_run_with_tools(self, tools):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
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
            response = component.run([ChatMessage.from_user("What's the weather like in Paris?")])

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

    def test_run_with_tools_streaming(self, mock_chat_completion_chunk_with_tools, tools):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        response = component.run(chat_messages, tools=tools)

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

    def test_invalid_tool_call_json(self, tools, caplog):
        caplog.set_level(logging.WARNING)

        with patch("openai.resources.chat.completions.Completions.create") as mock_create:
            mock_create.return_value = ChatCompletion(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="1",
                                    type="function",
                                    function=Function(name="weather", arguments='"invalid": "json"'),
                                )
                            ],
                        ),
                    )
                ],
                created=1234567890,
                usage=CompletionUsage(
                    completion_tokens=47,
                    prompt_tokens=540,
                    total_tokens=587,
                    completion_tokens_details=CompletionTokensDetails(
                        accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                    ),
                    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
                ),
            )

            component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"), tools=tools)
            response = component.run([ChatMessage.from_user("What's the weather in Paris?")])

        assert len(response["replies"]) == 1
        message = response["replies"][0]
        assert len(message.tool_calls) == 0
        assert "OpenAI returned a malformed JSON string for tool call arguments" in caplog.text
        assert message.meta["finish_reason"] == "tool_calls"
        assert message.meta["usage"]["completion_tokens"] == 47

    def test_convert_streaming_chunks_to_chat_message_tool_calls_in_any_chunk(self):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        chunk = chat_completion_chunk.ChatCompletionChunk(
            id="chatcmpl-B2g1XYv1WzALulC5c8uLtJgvEB48I",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None, function_call=None, refusal=None, role=None, tool_calls=None
                    ),
                    finish_reason="tool_calls",
                    index=0,
                    logprobs=None,
                )
            ],
            created=1739977895,
            model="gpt-4o-mini-2024-07-18",
            object="chat.completion.chunk",
            service_tier="default",
            system_fingerprint="fp_00428b782a",
            usage=None,
        )
        chunks = [
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": None,
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.910076",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id="call_ZOj5l67zhZOx6jqjg7ATQwb6",
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                arguments="", name="rag_pipeline_tool"
                            ),
                            type="function",
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.913919",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"qu', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.914439",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ery":', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.924146",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Wher', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.924420",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="e do", name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.944398",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="es Ma", name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.944958",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="rk liv", name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.945507",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='e?"}', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.946018",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=1,
                            id="call_STxsYY69wVOvxWqopAt3uWTB",
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                arguments="", name="get_weather"
                            ),
                            type="function",
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.946578",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=1,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"ci', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.946981",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=1,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ty": ', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.947411",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=1,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='"Berli', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.947643",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": [
                        chat_completion_chunk.ChoiceDeltaToolCall(
                            index=1,
                            id=None,
                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='n"}', name=None),
                            type=None,
                        )
                    ],
                    "finish_reason": None,
                    "received_at": "2025-02-19T16:02:55.947939",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "model": "gpt-4o-mini-2024-07-18",
                    "index": 0,
                    "tool_calls": None,
                    "finish_reason": "tool_calls",
                    "received_at": "2025-02-19T16:02:55.948772",
                },
            ),
        ]

        # Convert chunks to a chat message
        result = component._convert_streaming_chunks_to_chat_message(chunk, chunks)

        assert not result.texts
        assert not result.text

        # Verify both tool calls were found and processed
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_ZOj5l67zhZOx6jqjg7ATQwb6"
        assert result.tool_calls[0].tool_name == "rag_pipeline_tool"
        assert result.tool_calls[0].arguments == {"query": "Where does Mark live?"}
        assert result.tool_calls[1].id == "call_STxsYY69wVOvxWqopAt3uWTB"
        assert result.tool_calls[1].tool_name == "get_weather"
        assert result.tool_calls[1].arguments == {"city": "Berlin"}

        # Verify meta information
        assert result.meta["model"] == "gpt-4o-mini-2024-07-18"
        assert result.meta["finish_reason"] == "tool_calls"
        assert result.meta["index"] == 0
        assert result.meta["completion_start_time"] == "2025-02-19T16:02:55.910076"

    def test_convert_usage_chunk_to_streaming_chunk(self):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        chunk = ChatCompletionChunk(
            id="chatcmpl-BC1y4wqIhe17R8sv3lgLcWlB4tXCw",
            choices=[],
            created=1742207200,
            model="gpt-4o-mini-2024-07-18",
            object="chat.completion.chunk",
            service_tier="default",
            system_fingerprint="fp_06737a9306",
            usage=CompletionUsage(
                completion_tokens=8,
                prompt_tokens=13,
                total_tokens=21,
                completion_tokens_details=CompletionTokensDetails(
                    accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0
                ),
                prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
            ),
        )
        result = component._convert_chat_completion_chunk_to_streaming_chunk(chunk)
        assert result.content == ""
        assert result.meta["model"] == "gpt-4o-mini-2024-07-18"
        assert result.meta["received_at"] is not None

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIChatGenerator(generation_kwargs={"n": 1})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert message.meta["usage"]["prompt_tokens"] > 0

    async def test_run_with_wrong_model(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("Invalid model name")

        generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"), model="something-obviously-wrong")

        generator.client = mock_client

        with pytest.raises(OpenAIError):
            generator.run([ChatMessage.from_user("irrelevant")])

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = OpenAIChatGenerator(
            streaming_callback=callback, generation_kwargs={"stream_options": {"include_usage": True}}
        )
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

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
    def test_live_run_with_tools_streaming(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenAIChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(chat_messages)
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

    def test_openai_chat_generator_with_toolset_initialization(self, tools, monkeypatch):
        """Test that the OpenAIChatGenerator can be initialized with a Toolset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = OpenAIChatGenerator(tools=toolset)
        assert generator.tools == toolset

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        """Test that the OpenAIChatGenerator can be deserialized from a dictionary with a Toolset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = OpenAIChatGenerator(tools=toolset)
        data = component.to_dict()

        deserialized_component = OpenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_toolset(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        toolset = Toolset(tools)
        component = OpenAIChatGenerator(tools=toolset)
        results = component.run(chat_messages)
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_with_include_usage_serializes_nested_objects(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()

        component = OpenAIChatGenerator(
            streaming_callback=callback, generation_kwargs={"stream_options": {"include_usage": True}}
        )
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert "replies" in results
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "_meta" in message.__dict__

        metadata = message.meta
        assert "usage" in metadata
        usage = metadata["usage"]
        assert isinstance(usage, dict)
        assert "completion_tokens" in usage
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert "completion_tokens_details" in usage
        assert isinstance(usage["completion_tokens_details"], dict)
        assert "accepted_prediction_tokens" in usage["completion_tokens_details"]
        assert "prompt_tokens_details" in usage
        assert isinstance(usage["prompt_tokens_details"], dict)
        assert "audio_tokens" in usage["prompt_tokens_details"]
