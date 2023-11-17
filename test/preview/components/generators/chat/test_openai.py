import os
from unittest.mock import patch, Mock

import openai
import pytest

from haystack.preview.components.generators.chat import GPTChatGenerator
from haystack.preview.components.generators.utils import default_streaming_callback
from haystack.preview.dataclasses import ChatMessage, StreamingChunk


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.ChatCompletion.create", autospec=True) as mock_chat_completion_create:
        # mimic the response from the OpenAI API
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"

        mock_message = Mock()
        mock_message.content = "I'm fine, thanks. How are you?"
        mock_message.role = "user"

        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = Mock()
        mock_response.usage.items.return_value = [
            ("prompt_tokens", 57),
            ("completion_tokens", 40),
            ("total_tokens", 97),
        ]
        mock_response.choices = [mock_choice]
        mock_chat_completion_create.return_value = mock_response
        yield mock_chat_completion_create


def streaming_chunk(content: str):
    """
    Mock chunks of streaming responses from the OpenAI API
    """
    # mimic the chunk response from the OpenAI API
    mock_choice = Mock()
    mock_choice.index = 0
    mock_choice.delta.content = content
    mock_choice.finish_reason = "stop"

    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-3.5-turbo"
    mock_response.usage = Mock()
    mock_response.usage.items.return_value = [("prompt_tokens", 57), ("completion_tokens", 40), ("total_tokens", 97)]
    return mock_response


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class TestGPTChatGenerator:
    @pytest.mark.unit
    def test_init_default(self):
        component = GPTChatGenerator(api_key="test-api-key")
        assert openai.api_key == "test-api-key"
        assert component.model_name == "gpt-3.5-turbo"
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.openai.com/v1"
        assert openai.api_base == "https://api.openai.com/v1"
        assert not component.generation_kwargs

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        openai.api_key = None
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GPTChatGenerator expects an OpenAI API key"):
            GPTChatGenerator()

    @pytest.mark.unit
    def test_init_with_parameters(self):
        component = GPTChatGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        assert openai.api_key == "test-api-key"
        assert component.model_name == "gpt-4"
        assert component.streaming_callback is default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert openai.api_base == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_to_dict_default(self):
        component = GPTChatGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.chat.openai.GPTChatGenerator",
            "init_parameters": {
                "model_name": "gpt-3.5-turbo",
                "streaming_callback": None,
                "api_base_url": "https://api.openai.com/v1",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = GPTChatGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.chat.openai.GPTChatGenerator",
            "init_parameters": {
                "model_name": "gpt-4",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.utils.default_streaming_callback",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_lambda_streaming_callback(self):
        component = GPTChatGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.chat.openai.GPTChatGenerator",
            "init_parameters": {
                "model_name": "gpt-4",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "chat.test_openai.<lambda>",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.preview.components.generators.chat.openai.GPTChatGenerator",
            "init_parameters": {
                "model_name": "gpt-4",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.utils.default_streaming_callback",
            },
        }
        component = GPTChatGenerator.from_dict(data)
        assert component.model_name == "gpt-4"
        assert component.streaming_callback is default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        openai.api_key = None
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack.preview.components.generators.chat.openai.GPTChatGenerator",
            "init_parameters": {
                "model_name": "gpt-4",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.utils.default_streaming_callback",
            },
        }
        with pytest.raises(ValueError, match="GPTChatGenerator expects an OpenAI API key"):
            GPTChatGenerator.from_dict(data)

    @pytest.mark.unit
    def test_run(self, chat_messages, mock_chat_completion):
        component = GPTChatGenerator(api_key="test-api-key")
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = GPTChatGenerator(api_key="test-api-key", max_tokens=10, temperature=0.5)
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_run_streaming(self, chat_messages, mock_chat_completion):
        streaming_call_count = 0

        # Define the streaming callback function and assert that it is called with StreamingChunk objects
        def streaming_callback_fn(chunk: StreamingChunk):
            nonlocal streaming_call_count
            streaming_call_count += 1
            assert isinstance(chunk, StreamingChunk)

        generator = GPTChatGenerator(api_key="test-api-key", streaming_callback=streaming_callback_fn)

        # Create a fake streamed response
        # self needed here, don't remove
        def mock_iter(self):
            yield streaming_chunk("Hello")
            yield streaming_chunk("How are you?")

        mock_response = Mock(**{"__iter__": mock_iter})
        mock_chat_completion.return_value = mock_response

        response = generator.run(chat_messages)

        # Assert that the streaming callback was called twice
        assert streaming_call_count == 2

        # Assert that the response contains the generated replies
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.unit
    def test_check_abnormal_completions(self, caplog):
        component = GPTChatGenerator(api_key="test-api-key")
        messages = [
            ChatMessage.from_assistant(
                "", metadata={"finish_reason": "content_filter" if i % 2 == 0 else "length", "index": i}
            )
            for i, _ in enumerate(range(4))
        ]

        for m in messages:
            component._check_finish_reason(m)

        # check truncation warning
        message_template = (
            "The completion for index {index} has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

        for index in [1, 3]:
            assert caplog.records[index].message == message_template.format(index=index)

        # check content filter warning
        message_template = "The completion for index {index} has been truncated due to the content filter."
        for index in [0, 2]:
            assert caplog.records[index].message == message_template.format(index=index)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GPTChatGenerator(api_key=os.environ.get("OPENAI_API_KEY"), n=1)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert "gpt-3.5" in message.metadata["model"]
        assert message.metadata["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = GPTChatGenerator(model_name="something-obviously-wrong", api_key=os.environ.get("OPENAI_API_KEY"))
        with pytest.raises(openai.InvalidRequestError, match="The model `something-obviously-wrong` does not exist"):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = GPTChatGenerator(os.environ.get("OPENAI_API_KEY"), streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content

        assert "gpt-3.5" in message.metadata["model"]
        assert message.metadata["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses
