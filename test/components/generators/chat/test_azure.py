import pytest
from openai import OpenAIError

from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators.utils import default_streaming_callback


class TestOpenAIChatGenerator:
    def test_init_default(self):
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint", api_key="test-api-key")
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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

    # additional tests intentionally omitted as they are covered by test_openai.py
