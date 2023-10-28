from unittest.mock import patch, MagicMock, Mock

import pytest
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token, StreamDetails, FinishReason
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.preview.components.generators.hugging_face.hugging_face_remote import (
    HuggingFaceRemoteGenerator,
    ChatHuggingFaceRemoteGenerator,
)
from haystack.preview.dataclasses import StreamingChunk, ChatMessage


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


# used to test serialization of streaming_callback
def testing_streaming_callback(x):
    return x


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

        assert generator.model_id == model
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.tokenizer is None
        assert generator.client is not None
        assert generator.streaming_callback == streaming_callback

    @pytest.mark.unit
    def test_to_dict(self, mock_check_valid_model, mock_auto_tokenizer):
        # Initialize the HuggingFaceRemoteGenerator object with valid parameters
        generator = HuggingFaceRemoteGenerator(
            model="HuggingFaceH4/zephyr-7b-alpha",
            model_id="HuggingFaceH4/zephyr-7b-alpha",
            token="token",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=lambda x: x,
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["model"] == "HuggingFaceH4/zephyr-7b-alpha"
        assert init_params["model_id"] == "HuggingFaceH4/zephyr-7b-alpha"
        assert init_params["stop_words"] == ["stop", "words"]
        assert init_params["token"] is None
        assert init_params["generation_kwargs"] == {"n": 5, "stop_sequences": ["stop", "words"]}

    @pytest.mark.unit
    def test_serialize_and_deserialize(self, mock_check_valid_model, mock_auto_tokenizer):
        generator = HuggingFaceRemoteGenerator(
            model="HuggingFaceH4/zephyr-7b-alpha",
            model_id="HuggingFaceH4/zephyr-7b-alpha",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=testing_streaming_callback,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = HuggingFaceRemoteGenerator.from_dict(result)
        assert generator_2.model_id == "HuggingFaceH4/zephyr-7b-alpha"
        assert generator_2.generation_kwargs == {"n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback == testing_streaming_callback

    @pytest.mark.unit
    def test_initialize_with_url_for_model_without_model_id(self, mock_check_valid_model):
        # When model is a url, model_id must be provided
        model = "https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha"
        model_id = None

        with pytest.raises(ValueError):
            HuggingFaceRemoteGenerator(model=model, model_id=model_id)

        # but also if we provide invalid model_id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceRemoteGenerator(model=model, model_id="invalid_model_id")

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
        generator.warm_up()

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
        assert [isinstance(reply, str) for reply in response["replies"]]

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
        generator.warm_up()

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
        assert [isinstance(reply, str) for reply in response["replies"]]

        assert isinstance(response["metadata"], list)
        assert len(response["replies"]) == 3
        assert len(response["metadata"]) == 3
        assert [isinstance(reply, dict) for reply in response["metadata"]]

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
        generator.warm_up()

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
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0
        assert [isinstance(reply, dict) for reply in response["replies"]]

    @pytest.mark.unit
    def test_generate_text_with_custom_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        generator = HuggingFaceRemoteGenerator()
        generator.warm_up()

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run("How are you?", **generation_kwargs)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "max_new_tokens": 100, "stop_sequences": [], "temperature": 0.8}

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]
        assert response["replies"][0] == "I'm fine, thanks."

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

    @pytest.mark.unit
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
        generator = HuggingFaceRemoteGenerator(streaming_callback=streaming_callback_fn)
        generator.warm_up()

        # Create a fake streamed response
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
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": [], "stream": True}

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0
        assert [isinstance(reply, dict) for reply in response["replies"]]


chat_messages = [
    ChatMessage.from_system("You are a helpful assistant speaking on A2 level of English"),
    ChatMessage.from_user("Tell me about Berlin"),
]


class TestChatHuggingFaceRemoteGenerator:
    @pytest.mark.unit
    def test_initialize_with_valid_model_and_generation_parameters(self, mock_check_valid_model, mock_auto_tokenizer):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        model_id = None
        token = None
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = ChatHuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        assert generator.model_id == model
        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.tokenizer is not None
        assert generator.client is not None
        assert generator.streaming_callback == streaming_callback

    @pytest.mark.unit
    def test_to_dict(self, mock_check_valid_model, mock_auto_tokenizer):
        # Initialize the ChatHuggingFaceRemoteGenerator object with valid parameters
        generator = ChatHuggingFaceRemoteGenerator(
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
        assert init_params["model"] == "NousResearch/Llama-2-7b-chat-hf"
        assert init_params["model_id"] == "NousResearch/Llama-2-7b-chat-hf"
        assert init_params["stop_words"] == ["stop", "words"]
        assert init_params["token"] is None
        assert init_params["generation_kwargs"] == {"n": 5, "stop_sequences": ["stop", "words"]}

    @pytest.mark.unit
    def test_serialize_and_deserialize(self, mock_check_valid_model, mock_auto_tokenizer):
        generator = ChatHuggingFaceRemoteGenerator(
            model="HuggingFaceH4/zephyr-7b-alpha",
            model_id="HuggingFaceH4/zephyr-7b-alpha",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=testing_streaming_callback,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = ChatHuggingFaceRemoteGenerator.from_dict(result)
        assert generator_2.model_id == "HuggingFaceH4/zephyr-7b-alpha"
        assert generator_2.generation_kwargs == {"n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback == testing_streaming_callback

    @pytest.mark.unit
    def test_initialize_with_url_for_model_without_model_id(self, mock_check_valid_model):
        # When model is a url, model_id must be provided
        model = "https://some_chat_model.com"
        model_id = None

        with pytest.raises(ValueError):
            ChatHuggingFaceRemoteGenerator(model=model, model_id=model_id)

        # but also if we provide invalid model_id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            ChatHuggingFaceRemoteGenerator(model=model, model_id="invalid_model_id")

    @pytest.mark.unit
    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "meta-llama/Llama-2-13b-chat-hf"
        model_id = None
        token = None
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = ChatHuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        response = generator.run(messages=chat_messages)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_generate_multiple_text_responses_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        model = "meta-llama/Llama-2-13b-chat-hf"
        model_id = None
        token = None
        generation_kwargs = {"n": 3}
        stop_words = ["stop"]
        streaming_callback = None

        generator = ChatHuggingFaceRemoteGenerator(
            model=model,
            model_id=model_id,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        response = generator.run(chat_messages)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 3
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

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
            ChatHuggingFaceRemoteGenerator(
                model=model,
                model_id=model_id,
                token=token,
                generation_kwargs=generation_kwargs,
                stop_words=stop_words,
                streaming_callback=streaming_callback,
            )

    @pytest.mark.unit
    def test_generate_text_with_stop_words(self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation):
        generator = ChatHuggingFaceRemoteGenerator()
        generator.warm_up()

        stop_words = ["stop", "words"]

        # Generate text response with stop words
        response = generator.run(chat_messages, stop_words=stop_words)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop", "words"]}

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_generate_text_with_custom_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        generator = ChatHuggingFaceRemoteGenerator()
        generator.warm_up()

        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run(chat_messages, **generation_kwargs)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "max_new_tokens": 100, "stop_sequences": [], "temperature": 0.8}

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].content == "I'm fine, thanks."

    @pytest.mark.unit
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
        generator = ChatHuggingFaceRemoteGenerator(streaming_callback=streaming_callback_fn)
        generator.warm_up()

        # Create a fake streamed response
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
        response = generator.run(chat_messages)

        # check kwargs passed to text_generation
        args, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": [], "stream": True}

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
