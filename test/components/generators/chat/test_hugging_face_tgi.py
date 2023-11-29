from unittest.mock import patch, MagicMock, Mock

import pytest
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token, StreamDetails, FinishReason
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.generators.chat import HuggingFaceTGIChatGenerator

from haystack.dataclasses import StreamingChunk, ChatMessage


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.generators.chat.hugging_face_tgi.check_valid_model", MagicMock(return_value=None)
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


class TestHuggingFaceTGIChatGenerator:
    def test_initialize_with_valid_model_and_generation_parameters(self, mock_check_valid_model, mock_auto_tokenizer):
        model = "HuggingFaceH4/zephyr-7b-alpha"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        assert generator.generation_kwargs == {**generation_kwargs, **{"stop_sequences": ["stop"]}}
        assert generator.tokenizer is not None
        assert generator.client is not None
        assert generator.streaming_callback == streaming_callback

    def test_to_dict(self, mock_check_valid_model):
        # Initialize the HuggingFaceTGIChatGenerator object with valid parameters
        generator = HuggingFaceTGIChatGenerator(
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
        assert init_params["token"] is None
        assert init_params["generation_kwargs"] == {"n": 5, "stop_sequences": ["stop", "words"]}

    def test_from_dict(self, mock_check_valid_model):
        generator = HuggingFaceTGIChatGenerator(
            model="NousResearch/Llama-2-7b-chat-hf",
            generation_kwargs={"n": 5},
            stop_words=["stop", "words"],
            streaming_callback=streaming_callback_handler,
        )
        # Call the to_dict method
        result = generator.to_dict()

        generator_2 = HuggingFaceTGIChatGenerator.from_dict(result)
        assert generator_2.model == "NousResearch/Llama-2-7b-chat-hf"
        assert generator_2.generation_kwargs == {"n": 5, "stop_sequences": ["stop", "words"]}
        assert generator_2.streaming_callback is streaming_callback_handler

    def test_warm_up(self, mock_check_valid_model, mock_auto_tokenizer):
        generator = HuggingFaceTGIChatGenerator()
        generator.warm_up()

        # Assert that the tokenizer is now initialized
        assert generator.tokenizer is not None

    def test_warm_up_no_chat_template(self, mock_check_valid_model, mock_auto_tokenizer, caplog):
        generator = HuggingFaceTGIChatGenerator(model="meta-llama/Llama-2-13b-chat-hf")

        # Set chat_template to None for this specific test
        mock_auto_tokenizer.chat_template = None
        generator.warm_up()

        # warning message should be logged
        assert "The model 'meta-llama/Llama-2-13b-chat-hf' doesn't have a default chat_template" in caplog.text

    def test_custom_chat_template(
        self, chat_messages, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation
    ):
        custom_chat_template = "Here goes some Jinja template"

        # mocked method to check if we called apply_chat_template with the custom template
        mock_auto_tokenizer.apply_chat_template = MagicMock(return_value="some_value")

        generator = HuggingFaceTGIChatGenerator(chat_template=custom_chat_template)
        generator.warm_up()

        assert generator.chat_template == custom_chat_template

        generator.run(messages=chat_messages)
        assert mock_auto_tokenizer.apply_chat_template.call_count == 1

        # and we indeed called apply_chat_template with the custom template
        _, kwargs = mock_auto_tokenizer.apply_chat_template.call_args
        assert kwargs["chat_template"] == custom_chat_template

    def test_initialize_with_invalid_model_path_or_url(self, mock_check_valid_model):
        model = "invalid_model"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        mock_check_valid_model.side_effect = ValueError("Invalid model path or url")

        with pytest.raises(ValueError):
            HuggingFaceTGIChatGenerator(
                model=model,
                generation_kwargs=generation_kwargs,
                stop_words=stop_words,
                streaming_callback=streaming_callback,
            )

    def test_initialize_with_invalid_url(self, mock_check_valid_model):
        with pytest.raises(ValueError):
            HuggingFaceTGIChatGenerator(model="NousResearch/Llama-2-7b-chat-hf", url="invalid_url")

    def test_initialize_with_url_but_invalid_model(self, mock_check_valid_model):
        # When custom TGI endpoint is used via URL, model must be provided and valid HuggingFace Hub model id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceTGIChatGenerator(model="invalid_model_id", url="https://some_chat_model.com")

    def test_generate_text_response_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation, chat_messages
    ):
        model = "meta-llama/Llama-2-13b-chat-hf"
        generation_kwargs = {"n": 1}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIChatGenerator(
            model=model,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        response = generator.run(messages=chat_messages)

        # check kwargs passed to text_generation
        # note how n because it is not text generation parameter was not passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_generate_multiple_text_responses_with_valid_prompt_and_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation, chat_messages
    ):
        model = "meta-llama/Llama-2-13b-chat-hf"
        token = None
        generation_kwargs = {"n": 3}
        stop_words = ["stop"]
        streaming_callback = None

        generator = HuggingFaceTGIChatGenerator(
            model=model,
            token=token,
            generation_kwargs=generation_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )
        generator.warm_up()

        response = generator.run(chat_messages)

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop"]}

        # note how n caused n replies to be generated
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 3
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_generate_text_with_stop_words(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation, chat_messages
    ):
        generator = HuggingFaceTGIChatGenerator()
        generator.warm_up()

        stop_words = ["stop", "words"]

        # Generate text response with stop words
        response = generator.run(chat_messages, generation_kwargs={"stop_words": stop_words})

        # check kwargs passed to text_generation
        # we translate stop_words to stop_sequences
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": ["stop", "words"]}

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_generate_text_with_custom_generation_parameters(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation, chat_messages
    ):
        # Create an instance of HuggingFaceRemoteGenerator with no generation parameters
        generator = HuggingFaceTGIChatGenerator()
        generator.warm_up()

        # but then we pass them in run
        generation_kwargs = {"temperature": 0.8, "max_new_tokens": 100}
        response = generator.run(chat_messages, generation_kwargs=generation_kwargs)

        # again check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "max_new_tokens": 100, "stop_sequences": [], "temperature": 0.8}

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert response["replies"][0].content == "I'm fine, thanks."

    def test_generate_text_with_streaming_callback(
        self, mock_check_valid_model, mock_auto_tokenizer, mock_text_generation, chat_messages
    ):
        streaming_call_count = 0

        # Define the streaming callback function
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        # Create an instance of HuggingFaceRemoteGenerator
        generator = HuggingFaceTGIChatGenerator(streaming_callback=streaming_callback_fn)
        generator.warm_up()

        # Create a fake streamed response
        # self needed here, don't remove
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
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"details": True, "stop_sequences": [], "stream": True}

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
