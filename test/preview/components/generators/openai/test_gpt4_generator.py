from unittest.mock import patch

import pytest

from haystack.preview.components.generators.openai.gpt4 import GPT4Generator, default_streaming_callback


class TestChatGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self, caplog):
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = GPT4Generator()
            assert component.api_key is None
            assert component.model_name == "gpt-4"
            assert component.system_prompt == "You are a helpful assistant."
            assert component.max_tokens == 500
            assert component.temperature == 0.7
            assert component.top_p == 1
            assert component.n == 1
            assert component.stop == []
            assert component.presence_penalty == 0
            assert component.frequency_penalty == 0
            assert component.logit_bias == {}
            assert component.stream is False
            assert component.streaming_callback == default_streaming_callback
            assert component.streaming_done_marker == "[DONE]"
            assert component.api_base_url == "https://api.openai.com/v1"
            assert component.openai_organization is None
            assert component.max_tokens_limit == 8192

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert caplog.records[0].message == (
                "OpenAI API key is missing. You will need to provide an API key to Pipeline.run()."
            )

    @pytest.mark.unit
    def test_init_with_parameters(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.OPENAI_TOKENIZERS",
            {"test-model-name": "test-encoding"},
        )
        monkeypatch.setattr(
            "haystack.preview.components.generators.openai.chatgpt.OPENAI_TOKENIZERS_TOKEN_LIMITS",
            {"test-model-name": 10},
        )
        with patch("haystack.preview.components.generators.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = GPT4Generator(
                api_key="test-api-key",
                model_name="test-model-name",
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
                streaming_done_marker="test-marker",
                api_base_url="test-base-url",
                openai_organization="test-orga-id",
            )
            assert component.api_key == "test-api-key"
            assert component.model_name == "test-model-name"
            assert component.system_prompt == "test-system-prompt"
            assert component.max_tokens == 20
            assert component.temperature == 1
            assert component.top_p == 5
            assert component.n == 10
            assert component.stop == ["test-stop-word"]
            assert component.presence_penalty == 0.5
            assert component.frequency_penalty == 0.4
            assert component.logit_bias == {"test-logit-bias": 0.3}
            assert component.stream is True
            assert component.streaming_callback == callback
            assert component.streaming_done_marker == "test-marker"
            assert component.api_base_url == "test-base-url"
            assert component.openai_organization == "test-orga-id"
            assert component.max_tokens_limit == 10

            tiktoken_patch.get_encoding.assert_called_once_with("test-encoding")
            assert not caplog.records
