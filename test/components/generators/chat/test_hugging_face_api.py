# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from huggingface_hub import (
    ChatCompletionInputStreamOptions,
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
    ChatCompletionStreamOutputUsage,
)
from huggingface_hub.errors import RepositoryNotFoundError

from haystack import Pipeline
from haystack.components.generators.chat.hugging_face_api import (
    HuggingFaceAPIChatGenerator,
    _convert_chat_completion_stream_output_to_streaming_chunk,
    _convert_hfapi_tool_calls,
    _convert_tools_to_hfapi_tools,
)
from haystack.dataclasses import ChatMessage, ImageContent, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.tools.toolset import Toolset
from haystack.utils.auth import Secret
from haystack.utils.hf import HFGenerationAPIType


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]


def get_weather(city: str) -> dict[str, Any]:
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


@pytest.fixture
def tools():
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=get_weather,
    )
    return [weather_tool]


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.generators.chat.hugging_face_api.check_valid_model", MagicMock(return_value=None)
    ) as mock:
        yield mock


@pytest.fixture
def mock_chat_completion():
    # https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion.example

    with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock_chat_completion:
        completion = ChatCompletionOutput(
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason="eos_token",
                    index=0,
                    message=ChatCompletionOutputMessage(content="The capital of France is Paris.", role="assistant"),
                )
            ],
            id="some_id",
            model="some_model",
            system_fingerprint="some_fingerprint",
            usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
            created=1710498360,
        )

        mock_chat_completion.return_value = completion
        yield mock_chat_completion


@pytest.fixture
def mock_chat_completion_async():
    with patch("huggingface_hub.AsyncInferenceClient.chat_completion", autospec=True) as mock_chat_completion:
        completion = ChatCompletionOutput(
            choices=[
                ChatCompletionOutputComplete(
                    finish_reason="eos_token",
                    index=0,
                    message=ChatCompletionOutputMessage(content="The capital of France is Paris.", role="assistant"),
                )
            ],
            id="some_id",
            model="some_model",
            system_fingerprint="some_fingerprint",
            usage=ChatCompletionOutputUsage(completion_tokens=8, prompt_tokens=17, total_tokens=25),
            created=1710498360,
        )

        # Use AsyncMock to properly mock the async method
        mock_chat_completion.return_value = completion
        mock_chat_completion.__call__ = AsyncMock(return_value=completion)

        yield mock_chat_completion


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


class TestHuggingFaceAPIChatGenerator:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self, mock_check_valid_model):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": model},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator.api_params == {"model": model}
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop": ["stop"]}, **{"max_tokens": 512}}
        assert generator.streaming_callback == streaming_callback
        assert generator.tools is None

        # check that client and async_client are initialized
        assert generator._client.model == model
        assert generator._async_client.model == model

    def test_init_serverless_with_tools(self, mock_check_valid_model, tools):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": model},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
            tools=tools,
        )

        assert generator.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator.api_params == {"model": model}
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop": ["stop"]}, **{"max_tokens": 512}}
        assert generator.streaming_callback == streaming_callback
        assert generator.tools == tools

        assert generator._client.model == model
        assert generator._async_client.model == model

    def test_init_serverless_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
            )

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tgi(self):
        url = "https://some_model.com"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": url},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.TEXT_GENERATION_INFERENCE
        assert generator.api_params == {"url": url}
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop": ["stop"]}, **{"max_tokens": 512}}
        assert generator.streaming_callback == streaming_callback
        assert generator.tools is None

        assert generator._client.model == url
        assert generator._async_client.model == url

    def test_init_tgi_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tgi_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_init_fail_with_duplicate_tool_names(self, mock_check_valid_model, tools):
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "irrelevant"},
                tools=duplicate_tools,
            )

    def test_init_fail_with_tools_and_streaming(self, mock_check_valid_model, tools):
        with pytest.raises(ValueError):
            HuggingFaceAPIChatGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "irrelevant"},
                tools=tools,
                streaming_callback=streaming_callback_handler,
            )

    def test_to_dict(self, mock_check_valid_model):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )

        result = generator.to_dict()
        init_params = result["init_parameters"]

        assert init_params["api_type"] == "serverless_inference_api"
        assert init_params["api_params"] == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert init_params["token"] == {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        assert init_params["generation_kwargs"] == {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512}
        assert init_params["streaming_callback"] is None
        assert init_params["tools"] == [
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
        ]

    def test_from_dict(self, mock_check_valid_model):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )
        result = generator.to_dict()

        # now deserialize, call from_dict
        generator_2 = HuggingFaceAPIChatGenerator.from_dict(result)
        assert generator_2.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator_2.api_params == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert generator_2.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert generator_2.generation_kwargs == {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512}
        assert generator_2.streaming_callback is None
        assert generator_2.tools == [tool]

    def test_serde_in_pipeline(self, mock_check_valid_model):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack.components.generators.chat.hugging_face_api.HuggingFaceAPIChatGenerator",
                    "init_parameters": {
                        "api_type": "serverless_inference_api",
                        "api_params": {"model": "HuggingFaceH4/zephyr-7b-beta"},
                        "token": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": False},
                        "generation_kwargs": {"temperature": 0.6, "stop": ["stop", "words"], "max_tokens": 512},
                        "streaming_callback": None,
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "inputs_from_state": None,
                                    "name": "name",
                                    "outputs_to_state": None,
                                    "outputs_to_string": None,
                                    "description": "description",
                                    "parameters": {"x": {"type": "string"}},
                                    "function": "builtins.print",
                                },
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    def test_run(self, mock_check_valid_model, mock_chat_completion, chat_messages):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=None,
        )

        response = generator.run(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion.call_args
        hf_messages = [
            {"role": "system", "content": "You are a helpful assistant speaking A2 level of English"},
            {"role": "user", "content": "Tell me about Berlin"},
        ]
        assert kwargs == {
            "temperature": 0.6,
            "stop": ["stop", "words"],
            "max_tokens": 512,
            "tools": None,
            "messages": hf_messages,
        }

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_streaming_callback(self, mock_check_valid_model, mock_chat_completion, chat_messages):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            streaming_callback=streaming_callback_fn,
        )

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):
            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content="The", role="assistant"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content=None, role=None), index=0, finish_reason="length"
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_chat_completion.return_value = mock_response

        # Generate text response with streaming callback
        response = generator.run(chat_messages)

        # check kwargs passed to text_generation
        _, kwargs = mock_chat_completion.call_args
        assert kwargs == {
            "stop": [],
            "stream": True,
            "max_tokens": 512,
            "stream_options": ChatCompletionInputStreamOptions(include_usage=True),
        }

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_streaming_callback_in_run_method(
        self, mock_check_valid_model, mock_chat_completion, chat_messages
    ):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
        )

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):
            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content="The", role="assistant"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content=None, role=None), index=0, finish_reason="length"
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_chat_completion.return_value = mock_response

        # Generate text response with streaming callback
        response = generator.run(chat_messages, streaming_callback=streaming_callback_fn)

        # check kwargs passed to text_generation
        _, kwargs = mock_chat_completion.call_args
        assert kwargs == {
            "stop": [],
            "stream": True,
            "max_tokens": 512,
            "stream_options": ChatCompletionInputStreamOptions(include_usage=True),
        }

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_fail_with_tools_and_streaming(self, tools, mock_check_valid_model):
        component = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            streaming_callback=streaming_callback_handler,
        )

        with pytest.raises(ValueError):
            message = ChatMessage.from_user("irrelevant")
            component.run([message], tools=tools)

    def test_run_with_tools(self, mock_check_valid_model, tools):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-3.1-70B-Instruct"},
            tools=tools,
        )

        with patch("huggingface_hub.InferenceClient.chat_completion", autospec=True) as mock_chat_completion:
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    function=ChatCompletionOutputFunctionDefinition(
                                        arguments={"city": "Paris"}, name="weather", description=None
                                    ),
                                    id="0",
                                    type="function",
                                )
                            ],
                        ),
                        logprobs=None,
                    )
                ],
                created=1729074760,
                id="",
                model="meta-llama/Llama-3.1-70B-Instruct",
                system_fingerprint="2.3.2-dev0-sha-28bb7ae",
                usage=ChatCompletionOutputUsage(completion_tokens=30, prompt_tokens=426, total_tokens=456),
            )
            mock_chat_completion.return_value = completion

            messages = [ChatMessage.from_user("What is the weather in Paris?")]
            response = generator.run(messages=messages)

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].tool_calls[0].tool_name == "weather"
        assert response["replies"][0].tool_calls[0].arguments == {"city": "Paris"}
        assert response["replies"][0].tool_calls[0].id == "0"
        assert response["replies"][0].meta == {
            "finish_reason": "tool_calls",
            "index": 0,
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "usage": {"completion_tokens": 30, "prompt_tokens": 426},
        }

    def test_convert_hfapi_tool_calls_empty(self):
        hfapi_tool_calls = None
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 0

        hfapi_tool_calls = []
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 0

    def test_convert_hfapi_tool_calls_dict_arguments(self):
        hfapi_tool_calls = [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(
                    arguments={"city": "Paris"}, name="weather", description=None
                ),
                id="0",
                type="function",
            )
        ]
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "weather"
        assert tool_calls[0].arguments == {"city": "Paris"}
        assert tool_calls[0].id == "0"

    def test_convert_hfapi_tool_calls_str_arguments(self):
        hfapi_tool_calls = [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(
                    arguments='{"city": "Paris"}', name="weather", description=None
                ),
                id="0",
                type="function",
            )
        ]
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "weather"
        assert tool_calls[0].arguments == {"city": "Paris"}
        assert tool_calls[0].id == "0"

    def test_convert_hfapi_tool_calls_invalid_str_arguments(self):
        hfapi_tool_calls = [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(
                    arguments="not a valid JSON string", name="weather", description=None
                ),
                id="0",
                type="function",
            )
        ]
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 0

    def test_convert_hfapi_tool_calls_invalid_type_arguments(self):
        hfapi_tool_calls = [
            ChatCompletionOutputToolCall(
                function=ChatCompletionOutputFunctionDefinition(
                    arguments=["this", "is", "a", "list"], name="weather", description=None
                ),
                id="0",
                type="function",
            )
        ]
        tool_calls = _convert_hfapi_tool_calls(hfapi_tool_calls)
        assert len(tool_calls) == 0

    @pytest.mark.parametrize(
        "hf_stream_output, expected_stream_chunk, dummy_previous_chunks",
        [
            (
                ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role="assistant", content=" Paris"), index=0
                        )
                    ],
                    created=1748339326,
                    id="",
                    model="microsoft/Phi-3.5-mini-instruct",
                    system_fingerprint="3.2.1-sha-4d28897",
                ),
                StreamingChunk(
                    content=" Paris",
                    meta={
                        "received_at": "2025-05-27T12:14:28.228852",
                        "model": "microsoft/Phi-3.5-mini-instruct",
                        "finish_reason": None,
                    },
                    index=0,
                    start=True,
                ),
                [],
            ),
            (
                ChatCompletionStreamOutput(
                    choices=[
                        ChatCompletionStreamOutputChoice(
                            delta=ChatCompletionStreamOutputDelta(role="assistant", content=""),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    created=1748339326,
                    id="",
                    model="microsoft/Phi-3.5-mini-instruct",
                    system_fingerprint="3.2.1-sha-4d28897",
                ),
                StreamingChunk(
                    content="",
                    meta={
                        "received_at": "2025-05-27T12:14:28.228852",
                        "model": "microsoft/Phi-3.5-mini-instruct",
                        "finish_reason": "stop",
                    },
                    finish_reason="stop",
                ),
                [0],
            ),
            (
                ChatCompletionStreamOutput(
                    choices=[],
                    created=1748339326,
                    id="",
                    model="microsoft/Phi-3.5-mini-instruct",
                    system_fingerprint="3.2.1-sha-4d28897",
                    usage=ChatCompletionStreamOutputUsage(completion_tokens=2, prompt_tokens=21, total_tokens=23),
                ),
                StreamingChunk(
                    content="",
                    meta={
                        "received_at": "2025-05-27T12:14:28.228852",
                        "model": "microsoft/Phi-3.5-mini-instruct",
                        "usage": {"completion_tokens": 2, "prompt_tokens": 21},
                    },
                ),
                [0, 1],
            ),
        ],
    )
    def test_convert_chat_completion_stream_output_to_streaming_chunk(
        self, hf_stream_output, expected_stream_chunk, dummy_previous_chunks
    ):
        converted_stream_chunk = _convert_chat_completion_stream_output_to_streaming_chunk(
            chunk=hf_stream_output, previous_chunks=dummy_previous_chunks
        )
        # Remove timestamp from comparison since it's always the current time
        converted_stream_chunk.meta.pop("received_at", None)
        expected_stream_chunk.meta.pop("received_at", None)
        assert converted_stream_chunk == expected_stream_chunk

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    @pytest.mark.flaky(reruns=2, reruns_delay=10)
    def test_live_run_serverless(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen2.5-7B-Instruct", "provider": "together"},
            generation_kwargs={"max_tokens": 20},
        )

        # No need for instruction tokens here since we use the chat_completion endpoint which handles the chat
        # templating for us.
        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].text is not None
        meta = response["replies"][0].meta
        assert "usage" in meta
        assert "prompt_tokens" in meta["usage"]
        assert meta["usage"]["prompt_tokens"] > 0
        assert "completion_tokens" in meta["usage"]
        assert meta["usage"]["completion_tokens"] > 0
        assert meta["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert meta["finish_reason"] is not None

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    @pytest.mark.flaky(reruns=2, reruns_delay=10)
    def test_live_run_serverless_streaming(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen2.5-7B-Instruct", "provider": "together"},
            generation_kwargs={"max_tokens": 20},
            streaming_callback=streaming_callback_handler,
        )

        # No need for instruction tokens here since we use the chat_completion endpoint which handles the chat
        # templating for us.
        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        response = generator.run(messages=messages)

        print(response)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].text is not None

        response_meta = response["replies"][0].meta
        assert "completion_start_time" in response_meta
        assert datetime.fromisoformat(response_meta["completion_start_time"]) <= datetime.now()
        assert "usage" in response_meta
        assert "prompt_tokens" in response_meta["usage"]
        assert response_meta["usage"]["prompt_tokens"] >= 0
        assert "completion_tokens" in response_meta["usage"]
        assert response_meta["usage"]["completion_tokens"] >= 0
        # internally, Together calls this "Qwen/Qwen2.5-7B-Instruct-Turbo"
        assert "Qwen/Qwen2.5-7B-Instruct" in response_meta["model"]
        assert response_meta["finish_reason"] is not None

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_with_tools(self, tools):
        """
        We test the round trip: generate tool call, pass tool message, generate response.

        The model used here (Qwen/Qwen2.5-72B-Instruct) is not gated and kept in a warm state.
        """

        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen2.5-72B-Instruct", "provider": "together"},
            generation_kwargs={"temperature": 0.5},
        )

        results = generator.run(chat_messages, tools=tools)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert "city" in tool_call.arguments
        assert "Paris" in tool_call.arguments["city"]
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = chat_messages + [message, ChatMessage.from_tool(tool_result="22° C", origin=tool_call)]

        # the model tends to make tool calls if provided with tools, so we don't pass them here
        results = generator.run(new_messages, generation_kwargs={"max_tokens": 50})

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower() and "22" in final_message.text

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_multimodal(self, test_files_path):
        image_path = test_files_path / "images" / "apple.jpg"
        # Resize the image to keep this test fast
        image_content = ImageContent.from_file_path(file_path=image_path, size=(100, 100))
        messages = [ChatMessage.from_user(content_parts=["What does this image show? Max 5 words", image_content])]

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen2.5-VL-32B-Instruct", "provider": "fireworks-ai"},
            generation_kwargs={"max_tokens": 20},
        )

        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        message = response["replies"][0]
        assert message.text
        assert len(message.text) > 0
        assert any(word in message.text.lower() for word in ["apple", "fruit", "red"])

    @pytest.mark.asyncio
    async def test_run_async(self, mock_check_valid_model, mock_chat_completion_async, chat_messages):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=None,
        )

        response = await generator.run_async(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion_async.call_args
        hf_messages = [
            {"role": "system", "content": "You are a helpful assistant speaking A2 level of English"},
            {"role": "user", "content": "Tell me about Berlin"},
        ]
        assert kwargs == {
            "temperature": 0.6,
            "stop": ["stop", "words"],
            "max_tokens": 512,
            "tools": None,
            "messages": hf_messages,
        }

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_streaming(self, mock_check_valid_model, mock_chat_completion_async, chat_messages):
        streaming_call_count = 0

        async def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        # Create a fake streamed response
        async def mock_aiter(self):
            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content="The", role="assistant"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

            yield ChatCompletionStreamOutput(
                choices=[
                    ChatCompletionStreamOutputChoice(
                        delta=ChatCompletionStreamOutputDelta(content=None, role=None), index=0, finish_reason="length"
                    )
                ],
                id="some_id",
                model="some_model",
                system_fingerprint="some_fingerprint",
                created=1710498504,
            )

        mock_response = Mock(**{"__aiter__": mock_aiter})
        mock_chat_completion_async.return_value = mock_response

        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-2-13b-chat-hf"},
            streaming_callback=streaming_callback_fn,
        )

        response = await generator.run_async(messages=chat_messages)

        # check kwargs passed to chat_completion
        _, kwargs = mock_chat_completion_async.call_args
        assert kwargs == {
            "stop": [],
            "stream": True,
            "max_tokens": 512,
            "stream_options": ChatCompletionInputStreamOptions(include_usage=True),
        }

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_tools(self, tools, mock_check_valid_model):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "meta-llama/Llama-3.1-70B-Instruct"},
            tools=tools,
        )

        with patch("huggingface_hub.AsyncInferenceClient.chat_completion", autospec=True) as mock_chat_completion_async:
            completion = ChatCompletionOutput(
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    function=ChatCompletionOutputFunctionDefinition(
                                        arguments={"city": "Paris"}, name="weather", description=None
                                    ),
                                    id="0",
                                    type="function",
                                )
                            ],
                        ),
                        logprobs=None,
                    )
                ],
                created=1729074760,
                id="",
                model="meta-llama/Llama-3.1-70B-Instruct",
                system_fingerprint="2.3.2-dev0-sha-28bb7ae",
                usage=ChatCompletionOutputUsage(completion_tokens=30, prompt_tokens=426, total_tokens=456),
            )
            mock_chat_completion_async.return_value = completion

            messages = [ChatMessage.from_user("What is the weather in Paris?")]
            response = await generator.run_async(messages=messages)

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].tool_calls[0].tool_name == "weather"
        assert response["replies"][0].tool_calls[0].arguments == {"city": "Paris"}
        assert response["replies"][0].tool_calls[0].id == "0"
        assert response["replies"][0].meta == {
            "finish_reason": "tool_calls",
            "index": 0,
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "usage": {"completion_tokens": 30, "prompt_tokens": 426},
        }

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    @pytest.mark.flaky(reruns=2, reruns_delay=10)
    @pytest.mark.asyncio
    async def test_live_run_async_serverless(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "Qwen/Qwen2.5-7B-Instruct", "provider": "together"},
            generation_kwargs={"max_tokens": 20},
        )

        messages = [
            ChatMessage.from_user("What is the capital of France? Be concise only provide the capital, nothing else.")
        ]
        try:
            response = await generator.run_async(messages=messages)

            assert "replies" in response
            assert isinstance(response["replies"], list)
            assert len(response["replies"]) > 0
            assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
            assert response["replies"][0].text is not None

            meta = response["replies"][0].meta
            assert "usage" in meta
            assert "prompt_tokens" in meta["usage"]
            assert meta["usage"]["prompt_tokens"] > 0
            assert "completion_tokens" in meta["usage"]
            assert meta["usage"]["completion_tokens"] > 0
            assert meta["model"] == "Qwen/Qwen2.5-7B-Instruct"
            assert meta["finish_reason"] is not None
        finally:
            await generator._async_client.close()

    def test_hugging_face_api_generator_with_toolset_initialization(self, mock_check_valid_model, tools):
        """Test that the HuggingFaceAPIChatGenerator can be initialized with a Toolset."""
        toolset = Toolset(tools)
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "irrelevant"}, tools=toolset
        )
        assert generator.tools == toolset

    def test_from_dict_with_toolset(self, mock_check_valid_model, tools):
        """Test that the HuggingFaceAPIChatGenerator can be deserialized from a dictionary with a Toolset."""
        toolset = Toolset(tools)
        component = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "irrelevant"}, tools=toolset
        )
        data = component.to_dict()

        deserialized_component = HuggingFaceAPIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)

    def test_to_dict_with_toolset(self, mock_check_valid_model, tools):
        """Test that the HuggingFaceAPIChatGenerator can be serialized to a dictionary with a Toolset."""
        toolset = Toolset(tools[:1])
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "irrelevant"}, tools=toolset
        )
        data = generator.to_dict()

        expected_tools_data = {
            "type": "haystack.tools.toolset.Toolset",
            "data": {
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "weather",
                            "description": "useful to determine the weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                            "function": "generators.chat.test_hugging_face_api.get_weather",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    }
                ]
            },
        }
        assert data["init_parameters"]["tools"] == expected_tools_data

    def test_convert_tools_to_hfapi_tools(self):
        assert _convert_tools_to_hfapi_tools(None) is None
        assert _convert_tools_to_hfapi_tools([]) is None

        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=get_weather,
        )
        hf_tools = _convert_tools_to_hfapi_tools([tool])
        assert len(hf_tools) == 1
        assert hf_tools[0].type == "function"
        assert hf_tools[0].function.name == "weather"
        assert hf_tools[0].function.description == "useful to determine the weather in a given location"
        assert hf_tools[0].function.parameters == {"city": {"type": "string"}}

    def test_convert_tools_to_hfapi_tools_legacy(self):
        # this satisfies the check hasattr(ChatCompletionInputFunctionDefinition, "arguments")
        mock_class = MagicMock()

        with patch(
            "haystack.components.generators.chat.hugging_face_api.ChatCompletionInputFunctionDefinition", mock_class
        ):
            tool = Tool(
                name="weather",
                description="useful to determine the weather in a given location",
                parameters={"city": {"type": "string"}},
                function=get_weather,
            )
            _convert_tools_to_hfapi_tools([tool])

        mock_class.assert_called_once_with(
            name="weather",
            arguments={"city": {"type": "string"}},
            description="useful to determine the weather in a given location",
        )
