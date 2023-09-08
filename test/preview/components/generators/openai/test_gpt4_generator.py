from unittest.mock import patch, Mock

import pytest

from haystack.preview.components.generators.openai.gpt4 import GPT4Generator, API_BASE_URL
from haystack.preview.components.generators.openai.gpt35 import default_streaming_callback


class TestGPT4Generator:
    @pytest.mark.unit
    def test_init_default(self):
        component = GPT4Generator(api_key="test-api-key")
        assert component.system_prompt is None
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4"
        assert component.streaming_callback is None
        assert component.api_base_url == API_BASE_URL
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        callback = lambda x: x
        component = GPT4Generator(
            api_key="test-api-key",
            model_name="gpt-4-32k",
            system_prompt="test-system-prompt",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=callback,
            api_base_url="test-base-url",
        )
        assert component.system_prompt == "test-system-prompt"
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4-32k"
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_to_dict_default(self):
        component = GPT4Generator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "GPT4Generator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4",
                "system_prompt": None,
                "streaming_callback": None,
                "api_base_url": API_BASE_URL,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = GPT4Generator(
            api_key="test-api-key",
            model_name="gpt-4-32k",
            system_prompt="test-system-prompt",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "GPT4Generator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4-32k",
                "system_prompt": "test-system-prompt",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.openai.gpt35.default_streaming_callback",
            },
        }

    @pytest.mark.unit
    def test_from_dict_default(self):
        data = {"type": "GPT4Generator", "init_parameters": {"api_key": "test-api-key"}}
        component = GPT4Generator.from_dict(data)
        assert component.system_prompt is None
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4"
        assert component.streaming_callback is None
        assert component.api_base_url == API_BASE_URL
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "GPT4Generator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4-32k",
                "system_prompt": "test-system-prompt",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.openai.gpt35.default_streaming_callback",
            },
        }
        component = GPT4Generator.from_dict(data)
        assert component.system_prompt == "test-system-prompt"
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4-32k"
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}
