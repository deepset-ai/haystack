from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.preview.components.generators.hugging_face.hugging_face_remote import HuggingFaceRemoteGenerator


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.preview.components.generators.hugging_face.hugging_face_remote.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


@pytest.fixture
def mock_text_generation():
    with patch("huggingface_hub.InferenceClient.text_generation", autospec=True) as mock_from_pretrained:
        mock_response = Mock()
        mock_response.generated_text = "I'm fine, thanks."
        details = Mock()
        details.finish_reason = MagicMock(field1="value")
        details.tokens = [1, 2, 3]
        mock_response.details = details
        mock_from_pretrained.return_value = mock_response
        yield mock_from_pretrained


class TestHuggingFaceRemoteGenerator:
    @pytest.mark.unit
    def test_initialize_with_valid_model_and_generation_parameters(self, mock_check_valid_model, mock_auto_tokenizer):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        model_id = None
        token = None
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        assert generator.model_id == model_id
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.tokenizer is not None
        assert generator.client is not None
        assert generator.streaming_callback == streaming_callback

    @pytest.mark.unit
    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        model_id = None
        token = None
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert "metadata" in response
        assert isinstance(response["replies"], list)
        assert isinstance(response["metadata"], list)
        assert len(response["replies"]) == 1
        assert len(response["metadata"]) == 1

    @pytest.mark.unit
    def test_generate_multiple_text_responses_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        model_id = None
        token = None
        generation_kwargs = {"n": 3}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert "metadata" in response
        assert isinstance(response["replies"], list)
        assert isinstance(response["metadata"], list)
        assert len(response["replies"]) == 3
        assert len(response["metadata"]) == 3

    @pytest.mark.unit
    def test_initialize_with_invalid_model_path_or_url(self, mock_check_valid_model):
        model = "invalid_model"
        model_id = None
        token = None
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        mock_check_valid_model.side_effect = ValueError("Invalid model path or url")

        with pytest.raises(ValueError):
            HuggingFaceRemoteGenerator(
                model=model,
                model_id=model_id,
                token=token,
                generation_kwargs=generation_kwargs,
                stop_words=stop_words,
                streaming_callback=streaming_callback,
            )

    @pytest.mark.unit
    def test_generate_text_with_stop_words(self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation):
        generator = HuggingFaceRemoteGenerator()
        stop_words = ["stop", "words"]

        # Generate text response with stop words
        response = generator.run("How are you?", stop_words=stop_words)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop", "words"]}

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0

    @pytest.mark.unit
    def test_generate_text_with_custom_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        generator = HuggingFaceRemoteGenerator()
        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run("How are you?", **generation_kwargs)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "max_new_tokens": 100, "stop_sequences": [], "temperature": 0.8}

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert response["replies"][0] == "I'm fine, thanks."

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0

    @pytest.mark.skip(reason="Need to implement stream mocking")
    def test_generate_text_with_streaming_callback(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        # Create an instance of HuggingFaceRemoteGenerator
        generator = HuggingFaceRemoteGenerator()

        # Define the streaming callback function
        def streaming_callback(chunk):
            print(chunk.content)

        # Set the streaming callback function
        generator.streaming_callback = streaming_callback

        # Generate text response with streaming callback
        response = generator.run("prompt")

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0
