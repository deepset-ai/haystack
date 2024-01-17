from unittest.mock import patch, Mock

import pytest
from transformers import PreTrainedTokenizer

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


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
    mock_pipeline.tokenizer = mock_tokenizer

    return mock_pipeline


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
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text2text-generation", token="test-token"
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": "test-token",
        }

    def test_init_custom_device(self):
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text2text-generation", device="cuda:0"
        )

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
            "device": "cuda:0",
        }

    def test_init_task_parameter(self):
        generator = HuggingFaceLocalChatGenerator(task="text2text-generation")

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
        }

    def test_init_task_in_huggingface_pipeline_kwargs(self):
        generator = HuggingFaceLocalChatGenerator(huggingface_pipeline_kwargs={"task": "text2text-generation"})

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
        }

    def test_init_task_inferred_from_model_name(self, model_info_mock):
        generator = HuggingFaceLocalChatGenerator(model="mistralai/Mistral-7B-Instruct-v0.2")

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
        }

    def test_init_invalid_task(self):
        with pytest.raises(ValueError, match="is not supported."):
            HuggingFaceLocalChatGenerator(task="text-classification")

    def test_to_dict(self, model_info_mock):
        generator = HuggingFaceLocalChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            token="token",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=lambda x: x,
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["huggingface_pipeline_kwargs"]["model"] == "NousResearch/Llama-2-7b-chat-hf"
        assert "token" not in init_params["huggingface_pipeline_kwargs"]
        assert init_params["generation_kwargs"] == {"max_new_tokens": 512, "n": 5, "stop_sequences": ["stop", "words"]}

    def test_from_dict(self, model_info_mock):
        generator = HuggingFaceLocalChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=streaming_callback_handler,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = HuggingFaceLocalChatGenerator.from_dict(result)

        assert generator_2.generation_kwargs == {"max_new_tokens": 512, "n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback is streaming_callback_handler

    @patch("haystack.components.generators.chat.hugging_face_local.pipeline")
    def test_warm_up(self, pipeline_mock):
        generator = HuggingFaceLocalChatGenerator(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text2text-generation"
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="mistralai/Mistral-7B-Instruct-v0.2", task="text2text-generation", token=None
        )

    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, model_info_mock, mock_pipeline_tokenizer, chat_messages
    ):
        model = "meta-llama/Llama-2-13b-chat-hf"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceLocalChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        # Use the mocked pipeline from the fixture
        generator.pipeline = mock_pipeline_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.content == "Berlin is cool"
