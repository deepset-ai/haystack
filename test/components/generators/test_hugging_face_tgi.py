from unittest.mock import patch, MagicMock, Mock

import pytest
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token, StreamDetails, FinishReason
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.dataclasses import StreamingChunk


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.generators.hugging_face_tgi.check_valid_model", MagicMock(return_value=None)
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


class TestHuggingFaceTGIGenerator:
    def test_initialize_with_valid_model_and_generation_parameters(self, mock_check_valid_model):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIGenerator(
            model=model,
            url=None,
            token=None,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.model == model
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.tokenizer is None
        assert generator.client is not None
        assert generator.streaming_callback == streaming_callback

    def test_to_dict(self, mock_check_valid_model):
        # Initialize the HuggingFaceRemoteGenerator object with valid parameters
        generator = HuggingFaceTGIGenerator(
            token="token", generation_kwargs={"n": 5}, stop_words=["stop", "words"], streaming_callback=lambda x: x
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["model"] == "mistralai/Mistral-7B-v0.1"
        assert not init_params["token"]
        assert init_params["generation_kwargs"] == {"n": 5, "stop_sequences": ["stop", "words"]}

    def test_from_dict(self, mock_check_valid_model):
        generator = HuggingFaceTGIGenerator(
            model="mistralai/Mistral-7B-v0.1",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=streaming_callback_handler,
        )
        # Call the to_dict method
        result = generator.to_dict()

        # now deserialize, call from_dict
        generator_2 = HuggingFaceTGIGenerator.from_dict(result)
        assert generator_2.model == "mistralai/Mistral-7B-v0.1"
        assert generator_2.generation_kwargs == {"n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback is streaming_callback_handler

    def test_initialize_with_invalid_url(self, mock_check_valid_model):
        with pytest.raises(ValueError):
            HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", url="invalid_url")

    def test_initialize_with_url_but_invalid_model(self, mock_check_valid_model):
        # When custom TGI endpoint is used via URL, model must be provided and valid HuggingFace Hub model id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceTGIGenerator(model="invalid_model_id", url="https://some_chat_model.com")

    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "mistralai/Mistral-7B-v0.1"

        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert "meta" in response
        assert isinstance(response["replies"], list)
        assert isinstance(response["meta"], list)
        assert len(response["replies"]) == 1
        assert len(response["meta"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

    def test_generate_multiple_text_responses_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "mistralai/Mistral-7B-v0.1"
        generation_kwargs = {"n": 3}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert "meta" in response
        assert isinstance(response["replies"], list)
        assert [isinstance(reply, str) for reply in response["replies"]]

        assert isinstance(response["meta"], list)
        assert len(response["replies"]) == 3
        assert len(response["meta"]) == 3
        assert [isinstance(reply, dict) for reply in response["meta"]]

    def test_initialize_with_invalid_model(self, mock_check_valid_model):
        model = "invalid_model"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        mock_check_valid_model.side_effect = ValueError("Invalid model path or url")

        with pytest.raises(ValueError):
            HuggingFaceTGIGenerator(
                model=model,
                generation_kwargs=generation_kwargs,
                stop_words=stop_words,
                streaming_callback=streaming_callback,
            )

    def test_generate_text_with_stop_words(self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation):
        generator = HuggingFaceTGIGenerator()
        generator.warm_up()

        # Generate text response with stop words
        response = generator.run("How are you?", generation_kwargs={"stop_words": ["stop", "words"]})

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop", "words"]}

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Assert that the response contains the metadata
        assert "meta" in response
        assert isinstance(response["meta"], list)
        assert len(response["meta"]) > 0
        assert [isinstance(reply, dict) for reply in response["replies"]]

    def test_generate_text_with_custom_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        generator = HuggingFaceTGIGenerator()
        generator.warm_up()

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run("How are you?", generation_kwargs=generation_kwargs)

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "max_new_tokens": 100, "stop_sequences": [], "temperature": 0.8}

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

    def test_generate_text_with_streaming_callback(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        # Create an instance of HuggingFaceRemoteGenerator
        generator = HuggingFaceTGIGenerator(streaming_callback=streaming_callback_fn)
        generator.warm_up()

        # Create a fake streamed response
        # Don't remove self
        def mock_iter(self):
            yield TextGenerationStreamResponse(
                generated_text=None, token=Token(id=1, text="I'm fine, thanks.", logprob=0.0, special=False)
            )
            yield TextGenerationStreamResponse(
                generated_text=None,
                token=Token(id=1, text="Ok bye", logprob=0.0, special=False),
                details=StreamDetails(finish_reason=FinishReason.Length, generated_tokens=5),
            )

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_text_generation.return_value = mock_response

        # Generate text response with streaming callback
        response = generator.run("prompt")

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": [], "stream": True}

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
        assert [isinstance(reply, dict) for reply in response["replies"]]
