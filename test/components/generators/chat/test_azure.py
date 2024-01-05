import os

import pytest
from openai import OpenAIError

from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators.utils import default_streaming_callback
from haystack.dataclasses import ChatMessage


class TestOpenAIChatGenerator:
    def test_init_default(self):
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint", api_key="test-api-key")
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        with pytest.raises(OpenAIError):
            AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")

    def test_init_with_parameters(self):
        component = AzureOpenAIChatGenerator(
            azure_endpoint="some-non-existing-endpoint",
            api_key="test-api-key",
            streaming_callback=default_streaming_callback,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is default_streaming_callback
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self):
        component = AzureOpenAIChatGenerator(api_key="test-api-key", azure_endpoint="some-non-existing-endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-35-turbo",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self):
        component = AzureOpenAIChatGenerator(
            api_key="test-api-key",
            azure_endpoint="some-non-existing-endpoint",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-35-turbo",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) and not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AzureOpenAIChatGenerator(organization="HaystackCI")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert "gpt-35-turbo" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    # additional tests intentionally omitted as they are covered by test_openai.py
