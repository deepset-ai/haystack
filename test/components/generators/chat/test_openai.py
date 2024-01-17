import os

import pytest
from openai import OpenAIError

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import default_streaming_callback
from haystack.dataclasses import ChatMessage, StreamingChunk


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class TestOpenAIChatGenerator:
    def test_init_default(self):
        component = OpenAIChatGenerator(api_key="test-api-key")
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-3.5-turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAIError):
            OpenAIChatGenerator()

    def test_init_with_parameters(self):
        component = OpenAIChatGenerator(
            api_key="test-api-key",
            model="gpt-4",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4"
        assert component.streaming_callback is default_streaming_callback
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self):
        component = OpenAIChatGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "model": "gpt-3.5-turbo",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self):
        component = OpenAIChatGenerator(
            api_key="test-api-key",
            model="gpt-4",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "model": "gpt-4",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self):
        component = OpenAIChatGenerator(
            api_key="test-api-key",
            model="gpt-4",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "model": "gpt-4",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "chat.test_openai.<lambda>",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "model": "gpt-4",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = OpenAIChatGenerator.from_dict(data)
        assert component.model == "gpt-4"
        assert component.streaming_callback is default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "model": "gpt-4",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.default_streaming_callback",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(OpenAIError):
            OpenAIChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion):
        component = OpenAIChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = OpenAIChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
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

    def test_run_with_params_streaming(self, chat_messages, mock_chat_completion_chunk):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(streaming_callback=streaming_callback)
        response = component.run(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].content  # see mock_chat_completion_chunk

    def test_check_abnormal_completions(self, caplog):
        component = OpenAIChatGenerator(api_key="test-api-key")
        messages = [
            ChatMessage.from_assistant(
                "", meta={"finish_reason": "content_filter" if i % 2 == 0 else "length", "index": i}
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
        component = OpenAIChatGenerator(api_key=os.environ.get("OPENAI_API_KEY"), generation_kwargs={"n": 1})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert "gpt-3.5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = OpenAIChatGenerator(model="something-obviously-wrong", api_key=os.environ.get("OPENAI_API_KEY"))
        with pytest.raises(OpenAIError):
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
        component = OpenAIChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content

        assert "gpt-3.5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses
