from unittest.mock import patch

import pytest

from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator
from haystack.preview.components.generators.openai.chatgpt import default_streaming_callback
from haystack.preview.llm_backends.openai.chatgpt import ChatGPTBackend, DEFAULT_OPENAI_PARAMS


class TestChatGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            assert component.system_prompt is None
            assert component.llm.api_key is None
            assert component.llm.model_name == "gpt-3.5-turbo"
            assert component.llm.streaming_callback is None
            assert component.llm.api_base_url == "https://api.openai.com/v1"
            assert component.llm.model_parameters == DEFAULT_OPENAI_PARAMS
            assert isinstance(component.llm, ChatGPTBackend)

    @pytest.mark.unit
    def test_init_with_parameters(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="gpt-4",
                system_prompt="test-system-prompt",
                model_parameters={"max_tokens": 10, "some-test-param": "test-params"},
                streaming_callback=callback,
                api_base_url="test-base-url",
            )
            assert component.system_prompt == "test-system-prompt"
            assert component.llm.api_key == "test-api-key"
            assert component.llm.model_name == "gpt-4"
            assert component.llm.streaming_callback == callback
            assert component.llm.api_base_url == "test-base-url"
            assert component.llm.model_parameters == {
                **DEFAULT_OPENAI_PARAMS,
                "max_tokens": 10,
                "some-test-param": "test-params",
            }

    @pytest.mark.unit
    def test_to_dict_default(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            data = component.to_dict()
            assert data == {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": None,
                    "model_name": "gpt-3.5-turbo",
                    "system_prompt": None,
                    "model_parameters": DEFAULT_OPENAI_PARAMS,
                    "streaming_callback": None,
                    "api_base_url": "https://api.openai.com/v1",
                },
            }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="gpt-4",
                system_prompt="test-system-prompt",
                model_parameters={"max_tokens": 10, "some-test-params": "test-params"},
                streaming_callback=default_streaming_callback,
                api_base_url="test-base-url",
            )
            data = component.to_dict()
            assert data == {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": "test-api-key",
                    "model_name": "gpt-4",
                    "system_prompt": "test-system-prompt",
                    "model_parameters": {**DEFAULT_OPENAI_PARAMS, "max_tokens": 10, "some-test-params": "test-params"},
                    "api_base_url": "test-base-url",
                    "streaming_callback": "haystack.preview.components.generators.openai.chatgpt.default_streaming_callback",
                },
            }

    @pytest.mark.unit
    def test_from_dict(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            data = {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": "test-api-key",
                    "model_name": "gpt-4",
                    "system_prompt": "test-system-prompt",
                    "model_parameters": {"max_tokens": 10, "some-test-params": "test-params"},
                    "api_base_url": "test-base-url",
                    "streaming_callback": "haystack.preview.components.generators.openai.chatgpt.default_streaming_callback",
                },
            }
            component = ChatGPTGenerator.from_dict(data)
            assert component.system_prompt == "test-system-prompt"
            assert component.llm.api_key == "test-api-key"
            assert component.llm.model_name == "gpt-4"
            assert component.llm.streaming_callback == default_streaming_callback
            assert component.llm.api_base_url == "test-base-url"
            assert component.llm.model_parameters == {
                **DEFAULT_OPENAI_PARAMS,
                "max_tokens": 10,
                "some-test-params": "test-params",
            }

    @pytest.mark.unit
    def test_run_no_api_key(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            with pytest.raises(ValueError, match="OpenAI API key is missing. Please provide an API key."):
                component.run(prompts=["test"])

    @pytest.mark.unit
    def test_run_no_system_prompt(self):
        with patch("haystack.preview.components.generators.openai.chatgpt.ChatGPTBackend") as chatgpt_patch:
            chatgpt_patch.return_value.complete.side_effect = lambda chat, **kwargs: (
                [f"{msg.role}: {msg.content}" for msg in chat],
                {"some_info": None},
            )
            component = ChatGPTGenerator(api_key="test-api-key")
            results = component.run(prompts=["test-prompt-1", "test-prompt-2"])
            assert results == {
                "replies": [["user: test-prompt-1"], ["user: test-prompt-2"]],
                "metadata": [{"some_info": None}, {"some_info": None}],
            }

    @pytest.mark.unit
    def test_run_with_system_prompt(self):
        with patch("haystack.preview.components.generators.openai.chatgpt.ChatGPTBackend") as chatgpt_patch:
            chatgpt_patch.return_value.complete.side_effect = lambda chat, **kwargs: (
                [f"{msg.role}: {msg.content}" for msg in chat],
                {"some_info": None},
            )
            component = ChatGPTGenerator(api_key="test-api-key", system_prompt="test-system-prompt")
            results = component.run(prompts=["test-prompt-1", "test-prompt-2"])
            assert results == {
                "replies": [
                    ["system: test-system-prompt", "user: test-prompt-1"],
                    ["system: test-system-prompt", "user: test-prompt-2"],
                ],
                "metadata": [{"some_info": None}, {"some_info": None}],
            }
