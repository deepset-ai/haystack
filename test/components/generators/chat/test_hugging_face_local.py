# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch
from typing import Optional, List

from haystack.dataclasses.streaming_chunk import StreamingChunk
import pytest
from transformers import PreTrainedTokenizer

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.utils import ComponentDevice
from haystack.utils.auth import Secret
from haystack.tools import Tool


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"Weather data for {city}"


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant speaking A2 level of English"),
        ChatMessage.from_user("Tell me about Berlin"),
    ]


@pytest.fixture
def model_info_mock():
    with patch(
        "haystack.components.generators.chat.hugging_face_local.model_info",
        new=Mock(return_value=Mock(pipeline_tag="text2text-generation")),
    ) as mock:
        yield mock


@pytest.fixture
def mock_pipeline_tokenizer():
    # Mocking the pipeline
    mock_pipeline = Mock(return_value=[{"generated_text": "Berlin is cool"}])

    # Mocking the tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizer)
    mock_tokenizer.encode.return_value = ["Berlin", "is", "cool"]
    mock_tokenizer.pad_token_id = 100
    mock_pipeline.tokenizer = mock_tokenizer

    return mock_pipeline


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=get_weather,
    )

    return [tool]


def custom_tool_parser(text: str) -> Optional[List[ToolCall]]:
    """Test implementation of a custom tool parser."""
    return [ToolCall(tool_name="weather", arguments={"city": "Berlin"})]


class TestHuggingFaceLocalChatGenerator:
    def test_initialize_with_valid_model_and_generation_parameters(self, model_info_mock):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceLocalChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.streaming_callback == streaming_callback

    def test_init_custom_token(self):
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text2text-generation",
            token=Secret.from_token("test-token"),
            device=ComponentDevice.from_str("cpu"),
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": "test-token",
            "device": "cpu",
        }

    def test_init_custom_device(self):
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text2text-generation",
            device=ComponentDevice.from_str("cpu"),
            token=None,
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
            "device": "cpu",
        }

    def test_init_task_parameter(self):
        generator = HuggingFaceLocalChatGenerator(
            task="text2text-generation", device=ComponentDevice.from_str("cpu"), token=None
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text2text-generation",
            "token": None,
            "device": "cpu",
        }

    def test_init_task_in_huggingface_pipeline_kwargs(self):
        generator = HuggingFaceLocalChatGenerator(
            huggingface_pipeline_kwargs={"task": "text2text-generation"},
            device=ComponentDevice.from_str("cpu"),
            token=None,
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "task": "text2text-generation",
            "token": None,
            "device": "cpu",
        }

    def test_init_task_inferred_from_model_name(self, model_info_mock):
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", device=ComponentDevice.from_str("cpu"), token=None
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
            "device": "cpu",
        }

    def test_init_invalid_task(self):
        with pytest.raises(ValueError, match="is not supported."):
            HuggingFaceLocalChatGenerator(task="text-classification")

    def test_to_dict(self, model_info_mock, tools):
        generator = HuggingFaceLocalChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=None,
            chat_template="irrelevant",
            tools=tools,
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["token"] == {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"}
        assert init_params["huggingface_pipeline_kwargs"]["model"] == "NousResearch/Llama-2-7b-chat-hf"
        assert "token" not in init_params["huggingface_pipeline_kwargs"]
        assert init_params["generation_kwargs"] == {"max_new_tokens": 512, "n": 5, "stop_sequences": ["stop", "words"]}
        assert init_params["streaming_callback"] is None
        assert init_params["chat_template"] == "irrelevant"
        assert init_params["tools"] == [
            {
                "type": "haystack.tools.tool.Tool",
                "data": {
                    "name": "weather",
                    "description": "useful to determine the weather in a given location",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    "function": "chat.test_hugging_face_local.get_weather",
                },
            }
        ]

    def test_from_dict(self, model_info_mock, tools):
        generator = HuggingFaceLocalChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=None,
            chat_template="irrelevant",
            tools=tools,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = HuggingFaceLocalChatGenerator.from_dict(result)

        assert generator_2.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert generator_2.generation_kwargs == {"max_new_tokens": 512, "n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback is None
        assert generator_2.chat_template == "irrelevant"
        assert len(generator_2.tools) == 1
        assert generator_2.tools[0].name == "weather"
        assert generator_2.tools[0].description == "useful to determine the weather in a given location"
        assert generator_2.tools[0].parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

    @patch("haystack.components.generators.chat.hugging_face_local.pipeline")
    def test_warm_up(self, pipeline_mock, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="text2text-generation",
            device=ComponentDevice.from_str("cpu"),
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text2text-generation", token=None, device="cpu"
        )

    def test_run(self, model_info_mock, mock_pipeline_tokenizer, chat_messages):
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"

    def test_run_with_custom_generation_parameters(self, model_info_mock, mock_pipeline_tokenizer, chat_messages):
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer
        results = generator.run(messages=chat_messages, generation_kwargs=generation_kwargs)

        # check kwargs passed pipeline
        _, kwargs = generator.pipeline.call_args
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == 0.8

        # replies are properly parsed and returned
        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"

    def test_run_with_streaming_callback(self, model_info_mock, mock_pipeline_tokenizer, chat_messages):
        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk): ...

        generator = HuggingFaceLocalChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf", streaming_callback=streaming_callback_fn
        )

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"
        generator.pipeline.assert_called_once()
        generator.pipeline.call_args[1]["streamer"].token_handler == streaming_callback_fn

    def test_run_with_streaming_callback_in_run_method(self, model_info_mock, mock_pipeline_tokenizer, chat_messages):
        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk): ...

        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer

        results = generator.run(messages=chat_messages, streaming_callback=streaming_callback_fn)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.text == "Berlin is cool"
        generator.pipeline.assert_called_once()
        generator.pipeline.call_args[1]["streamer"].token_handler == streaming_callback_fn

    @patch("haystack.components.generators.chat.hugging_face_local.convert_message_to_hf_format")
    def test_messages_conversion_is_called(self, mock_convert, model_info_mock):
        generator = HuggingFaceLocalChatGenerator(model="fake-model")

        messages = [ChatMessage.from_user("Hello"), ChatMessage.from_assistant("Hi there")]

        with patch.object(generator, "pipeline") as mock_pipeline:
            mock_pipeline.tokenizer.apply_chat_template.return_value = "test prompt"
            mock_pipeline.return_value = [{"generated_text": "test response"}]

            generator.warm_up()
            generator.run(messages)

        assert mock_convert.call_count == 2
        mock_convert.assert_any_call(messages[0])
        mock_convert.assert_any_call(messages[1])

    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=10)
    def test_live_run(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)  # https://github.com/deepset-ai/haystack/issues/8811
        messages = [ChatMessage.from_user("Please create a summary about the following topic: Climate change")]

        llm = HuggingFaceLocalChatGenerator(
            model="Qwen/Qwen2.5-0.5B-Instruct", generation_kwargs={"max_new_tokens": 50}
        )
        llm.warm_up()

        result = llm.run(messages)

        assert "replies" in result
        assert isinstance(result["replies"][0], ChatMessage)
        assert "climate change" in result["replies"][0].text.lower()

    def test_init_fail_with_duplicate_tool_names(self, model_info_mock, tools):
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError, match="Duplicate tool names found"):
            HuggingFaceLocalChatGenerator(model="irrelevant", tools=duplicate_tools)

    def test_init_fail_with_tools_and_streaming(self, model_info_mock, tools):
        with pytest.raises(ValueError, match="Using tools and streaming at the same time is not supported"):
            HuggingFaceLocalChatGenerator(
                model="irrelevant", tools=tools, streaming_callback=streaming_callback_handler
            )

    def test_run_with_tools(self, model_info_mock, tools):
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf", tools=tools)

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Paris"}}'}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_run_with_tools_in_run_method(self, model_info_mock, tools):
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Paris"}}'}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        results = generator.run(messages=messages, tools=tools)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.tool_calls
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_run_with_tools_and_tool_response(self, model_info_mock, tools):
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Mock pipeline and tokenizer
        mock_pipeline = Mock(return_value=[{"generated_text": "The weather in Paris is 22°C"}])
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.encode.return_value = ["some", "tokens"]
        mock_tokenizer.pad_token_id = 100
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_pipeline.tokenizer = mock_tokenizer
        generator.pipeline = mock_pipeline

        tool_call = ToolCall(tool_name="weather", arguments={"city": "Paris"})
        messages = [
            ChatMessage.from_user("What's the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="22°C", origin=tool_call),
        ]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert not message.tool_calls  # No tool calls in the final response
        assert "22°C" in message.text
        assert message.meta["finish_reason"] == "stop"

    def test_run_with_custom_tool_parser(self, model_info_mock, tools):
        """Test that a custom tool parsing function works correctly."""
        generator = HuggingFaceLocalChatGenerator(
            model="meta-llama/Llama-2-13b-chat-hf", tools=tools, tool_parsing_function=custom_tool_parser
        )
        generator.pipeline = Mock(return_value=[{"mocked_response": "Mocked response, we don't use it"}])
        generator.pipeline.tokenizer = Mock()
        generator.pipeline.tokenizer.encode.return_value = [1, 2, 3]
        generator.pipeline.tokenizer.pad_token_id = 1

        messages = [ChatMessage.from_user("What's the weather like in Berlin?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        assert len(results["replies"][0].tool_calls) == 1
        assert results["replies"][0].tool_calls[0].tool_name == "weather"
        assert results["replies"][0].tool_calls[0].arguments == {"city": "Berlin"}

    def test_default_tool_parser(self, model_info_mock, tools):
        """Test that the default tool parser works correctly with valid tool call format."""
        generator = HuggingFaceLocalChatGenerator(model="meta-llama/Llama-2-13b-chat-hf", tools=tools)
        generator.pipeline = Mock(
            return_value=[{"generated_text": '{"name": "weather", "arguments": {"city": "Berlin"}}'}]
        )
        generator.pipeline.tokenizer = Mock()
        generator.pipeline.tokenizer.encode.return_value = [1, 2, 3]
        generator.pipeline.tokenizer.pad_token_id = 1

        messages = [ChatMessage.from_user("What's the weather like in Berlin?")]
        results = generator.run(messages=messages)

        assert len(results["replies"]) == 1
        assert len(results["replies"][0].tool_calls) == 1
        assert results["replies"][0].tool_calls[0].tool_name == "weather"
        assert results["replies"][0].tool_calls[0].arguments == {"city": "Berlin"}
