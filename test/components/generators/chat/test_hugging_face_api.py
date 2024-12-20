# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack.utils.hf import HFGenerationAPIType
from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
)
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.generators.chat.hugging_face_api import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage, Tool, ToolCall


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]


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
                "description": "description",
                "function": "builtins.print",
                "name": "name",
                "parameters": {"x": {"type": "string"}},
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
                                "name": "name",
                                "description": "description",
                                "parameters": {"x": {"type": "string"}},
                                "function": "builtins.print",
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
        assert kwargs == {"stop": [], "stream": True, "max_tokens": 512}

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
            "finish_reason": "stop",
            "index": 0,
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "usage": {"completion_tokens": 30, "prompt_tokens": 426},
        }

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_serverless(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"max_tokens": 20},
        )

        messages = [ChatMessage.from_user("What is the capital of France?")]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "usage" in response["replies"][0].meta
        assert "prompt_tokens" in response["replies"][0].meta["usage"]
        assert "completion_tokens" in response["replies"][0].meta["usage"]

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_serverless_streaming(self):
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"max_tokens": 20},
            streaming_callback=streaming_callback_handler,
        )

        messages = [ChatMessage.from_user("What is the capital of France?")]
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "usage" in response["replies"][0].meta
        assert "prompt_tokens" in response["replies"][0].meta["usage"]
        assert "completion_tokens" in response["replies"][0].meta["usage"]

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        """
        We test the round trip: generate tool call, pass tool message, generate response.

        The model used here (zephyr-7b-beta) is always available and not gated.
        Even if it does not officially support tools, TGI+HF API make it work.
        """

        chat_messages = [ChatMessage.from_user("What's the weather like in Paris and Munich?")]
        generator = HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
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
        assert message.meta["finish_reason"] == "stop"

        new_messages = chat_messages + [message, ChatMessage.from_tool(tool_result="22° C", origin=tool_call)]

        # the model tends to make tool calls if provided with tools, so we don't pass them here
        results = generator.run(new_messages, generation_kwargs={"max_tokens": 50})

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
