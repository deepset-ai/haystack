# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from huggingface_hub import (
    TextGenerationOutputToken,
    TextGenerationStreamOutput,
    TextGenerationStreamOutputStreamDetails,
)
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack.utils.hf import HFGenerationAPIType


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.generators.hugging_face_api.check_valid_model", MagicMock(return_value=None)
    ) as mock:
        yield mock


@pytest.fixture
def mock_text_generation():
    with patch("huggingface_hub.InferenceClient.text_generation", autospec=True) as mock_text_generation:
        mock_response = Mock()
        mock_response.generated_text = "I'm fine, thanks."
        details = Mock()
        details.finish_reason = MagicMock(field1="value")
        details.tokens = [1, 2, 3]
        mock_response.details = details
        mock_text_generation.return_value = mock_response
        yield mock_text_generation


# used to test serialization of streaming_callback
def streaming_callback_handler(x):
    return x


class TestHuggingFaceAPIGenerator:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIGenerator(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self, mock_check_valid_model):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": model},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator.api_params == {"model": model}
        assert generator.generation_kwargs == {
            **generation_kwargs,
            **{"stop_sequences": ["stop"]},
            **{"max_new_tokens": 512},
        }
        assert generator.streaming_callback == streaming_callback

    def test_init_serverless_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceAPIGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
            )

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIGenerator(
                api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tgi(self):
        url = "https://some_model.com"
        generation_kwargs = {"temperature": 0.6}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE,
            api_params={"url": url},
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.api_type == HFGenerationAPIType.TEXT_GENERATION_INFERENCE
        assert generator.api_params == {"url": url}
        assert generator.generation_kwargs == {
            **generation_kwargs,
            **{"stop_sequences": ["stop"]},
            **{"max_new_tokens": 512},
        }
        assert generator.streaming_callback == streaming_callback

    def test_init_tgi_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tgi_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIGenerator(
                api_type=HFGenerationAPIType.TEXT_GENERATION_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_to_dict(self, mock_check_valid_model):
        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
        )

        result = generator.to_dict()
        init_params = result["init_parameters"]

        assert init_params["api_type"] == "serverless_inference_api"
        assert init_params["api_params"] == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert init_params["token"] == {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        assert init_params["generation_kwargs"] == {
            "temperature": 0.6,
            "stop_sequences": ["stop", "words"],
            "max_new_tokens": 512,
        }

    def test_from_dict(self, mock_check_valid_model):
        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=streaming_callback_handler,
        )
        result = generator.to_dict()

        # now deserialize, call from_dict
        generator_2 = HuggingFaceAPIGenerator.from_dict(result)
        assert generator_2.api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert generator_2.api_params == {"model": "HuggingFaceH4/zephyr-7b-beta"}
        assert generator_2.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert generator_2.generation_kwargs == {
            "temperature": 0.6,
            "stop_sequences": ["stop", "words"],
            "max_new_tokens": 512,
        }
        assert generator_2.streaming_callback is streaming_callback_handler

    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_text_generation
    ):
        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            token=Secret.from_env_var("ENV_VAR", strict=False),
            generation_kwargs={"temperature": 0.6},
            stop_words=["stop", "words"],
            streaming_callback=None,
        )

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {
            "details": True,
            "temperature": 0.6,
            "stop_sequences": ["stop", "words"],
            "stream": False,
            "max_new_tokens": 512,
        }

        assert isinstance(response, dict)
        assert "replies" in response
        assert "meta" in response
        assert isinstance(response["replies"], list)
        assert isinstance(response["meta"], list)
        assert len(response["replies"]) == 1
        assert len(response["meta"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

    def test_generate_text_with_custom_generation_parameters(self, mock_check_valid_model, mock_text_generation):
        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "HuggingFaceH4/zephyr-7b-beta"}
        )

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run("How are you?", generation_kwargs=generation_kwargs)

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {
            "details": True,
            "max_new_tokens": 100,
            "stop_sequences": [],
            "stream": False,
            "temperature": 0.8,
        }

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]
        assert response["replies"][0] == "I'm fine, thanks."

        # Assert that the response contains the metadata
        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

    def test_generate_text_with_streaming_callback(self, mock_check_valid_model, mock_text_generation):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            streaming_callback=streaming_callback_fn,
        )

        # Create a fake streamed response
        # Don't remove self
        def mock_iter(self):
            yield TextGenerationStreamOutput(
                index=0,
                generated_text=None,
                token=TextGenerationOutputToken(id=1, text="I'm fine, thanks.", logprob=0.0, special=False),
            )
            yield TextGenerationStreamOutput(
                index=1,
                generated_text=None,
                token=TextGenerationOutputToken(id=1, text="Ok bye", logprob=0.0, special=False),
                details=TextGenerationStreamOutputStreamDetails(
                    finish_reason="length", generated_tokens=5, seed=None, input_length=10
                ),
            )

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_text_generation.return_value = mock_response

        # Generate text response with streaming callback
        response = generator.run("prompt")

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": [], "stream": True, "max_new_tokens": 512}

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Assert that the response contains the metadata
        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) > 0
        assert [isinstance(meta, dict) for meta in response["meta"]]

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_run_serverless(self):
        generator = HuggingFaceAPIGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
            generation_kwargs={"max_new_tokens": 20},
        )

        response = generator.run("How are you?")
        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Assert that the response contains the metadata
        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) > 0
        assert [isinstance(meta, dict) for meta in response["meta"]]
