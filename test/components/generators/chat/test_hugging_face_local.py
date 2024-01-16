from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


class TestHuggingFaceLocalChatGenerator:
    @patch("haystack.components.generators.chat.hugging_face_local.model_info")
    def test_initialize_with_valid_model_and_generation_parameters(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
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

    @patch("haystack.components.generators.chat.hugging_face_local.model_info")
    def test_init_task_inferred_from_model_name(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        generator = HuggingFaceLocalChatGenerator(model="mistralai/Mistral-7B-Instruct-v0.2")

        assert generator.huggingface_pipeline_kwargs == {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text2text-generation",
            "token": None,
        }

    def test_init_invalid_task(self):
        with pytest.raises(ValueError, match="is not supported."):
            HuggingFaceLocalChatGenerator(task="text-classification")

    @patch("haystack.components.generators.chat.hugging_face_local.model_info")
    def test_to_dict(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
        # Initialize the HuggingFaceLocalChatGenerator object with valid parameters
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

    @patch("haystack.components.generators.chat.hugging_face_local.model_info")
    def test_from_dict(self, model_info_mock):
        model_info_mock.return_value.pipeline_tag = "text2text-generation"
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
