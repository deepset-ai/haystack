from unittest.mock import patch

import pytest

from haystack.preview.components.generators.openai.chatgpt import ChatGPTGenerator
from haystack.preview.components.generators.openai.chatgpt import default_streaming_callback


class TestChatGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTGenerator()
            assert component.system_prompt is None
            assert component.llm.api_key is None
            assert component.llm.model_name == "gpt-3.5-turbo"
            assert component.llm.max_tokens == 500
            assert component.llm.temperature == 0.7
            assert component.llm.top_p == 1
            assert component.llm.n == 1
            assert component.llm.stop == []
            assert component.llm.presence_penalty == 0
            assert component.llm.frequency_penalty == 0
            assert component.llm.logit_bias == {}
            assert component.llm.stream is False
            assert component.llm.streaming_callback == default_streaming_callback
            assert component.llm.api_base_url == "https://api.openai.com/v1"
            assert component.llm.openai_organization is None
            assert component.llm.max_tokens_limit == 4097

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert caplog.records[0].message == (
                "OpenAI API key is missing. You will need to provide an API key to Pipeline.run()."
            )

    @pytest.mark.unit
    def test_init_with_parameters(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="gpt-4",
                system_prompt="test-system-prompt",
                max_tokens=20,
                temperature=1,
                top_p=5,
                n=10,
                stop=["test-stop-word"],
                presence_penalty=0.5,
                frequency_penalty=0.4,
                logit_bias={"test-logit-bias": 0.3},
                stream=True,
                streaming_callback=callback,
                api_base_url="test-base-url",
                openai_organization="test-orga-id",
            )
            assert component.system_prompt == "test-system-prompt"
            assert component.llm.api_key == "test-api-key"
            assert component.llm.model_name == "gpt-4"
            assert component.llm.max_tokens == 20
            assert component.llm.temperature == 1
            assert component.llm.top_p == 5
            assert component.llm.n == 10
            assert component.llm.stop == ["test-stop-word"]
            assert component.llm.presence_penalty == 0.5
            assert component.llm.frequency_penalty == 0.4
            assert component.llm.logit_bias == {"test-logit-bias": 0.3}
            assert component.llm.stream is True
            assert component.llm.streaming_callback == callback
            assert component.llm.api_base_url == "test-base-url"
            assert component.llm.openai_organization == "test-orga-id"
            assert component.llm.max_tokens_limit == 8192

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert not caplog.records

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
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 1,
                    "n": 1,
                    "stop": [],
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "logit_bias": {},
                    "stream": False,
                    # FIXME serialize callback?
                    "api_base_url": "https://api.openai.com/v1",
                    "openai_organization": None,
                },
            }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTGenerator(
                api_key="test-api-key",
                model_name="gpt-4",
                system_prompt="test-system-prompt",
                max_tokens=20,
                temperature=1,
                top_p=5,
                n=10,
                stop=["test-stop-word"],
                presence_penalty=0.5,
                frequency_penalty=0.4,
                logit_bias={"test-logit-bias": 0.3},
                stream=True,
                streaming_callback=callback,
                api_base_url="test-base-url",
                openai_organization="test-orga-id",
            )
            data = component.to_dict()
            assert data == {
                "type": "ChatGPTGenerator",
                "init_parameters": {
                    "api_key": "test-api-key",
                    "model_name": "gpt-4",
                    "system_prompt": "test-system-prompt",
                    "max_tokens": 20,
                    "temperature": 1,
                    "top_p": 5,
                    "n": 10,
                    "stop": ["test-stop-word"],
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.4,
                    "logit_bias": {"test-logit-bias": 0.3},
                    "stream": True,
                    # FIXME serialize callback?
                    "api_base_url": "test-base-url",
                    "openai_organization": "test-orga-id",
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
                    "max_tokens": 20,
                    "temperature": 1,
                    "top_p": 5,
                    "n": 10,
                    "stop": ["test-stop-word"],
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.4,
                    "logit_bias": {"test-logit-bias": 0.3},
                    "stream": True,
                    # FIXME serialize callback?
                    "api_base_url": "test-base-url",
                    "openai_organization": "test-orga-id",
                },
            }
            component = ChatGPTGenerator.from_dict(data)
            assert component.system_prompt == "test-system-prompt"
            assert component.llm.api_key == "test-api-key"
            assert component.llm.model_name == "gpt-4"
            assert component.llm.max_tokens == 20
            assert component.llm.temperature == 1
            assert component.llm.top_p == 5
            assert component.llm.n == 10
            assert component.llm.stop == ["test-stop-word"]
            assert component.llm.presence_penalty == 0.5
            assert component.llm.frequency_penalty == 0.4
            assert component.llm.logit_bias == {"test-logit-bias": 0.3}
            assert component.llm.stream is True
            assert component.llm.streaming_callback == default_streaming_callback
            assert component.llm.api_base_url == "test-base-url"
            assert component.llm.openai_organization == "test-orga-id"
            assert component.llm.max_tokens_limit == 8192

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
