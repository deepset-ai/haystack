from unittest.mock import patch, Mock

import pytest

from haystack.preview.llm_backends.openai.chatgpt import ChatGPTBackend, ChatMessage


class TestChatGPTBackend:
    @pytest.mark.unit
    def test_init_default(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTBackend()
            assert component.api_key is None
            assert component.model_name == "gpt-3.5-turbo"
            assert component.model_parameters == {
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 1,
                "n": 1,
                "stop": [],
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "logit_bias": {},
                "stream": False,
                "openai_organization": None,
            }
            assert component.streaming_callback is None
            assert component.api_base_url == "https://api.openai.com/v1"
            assert component.max_tokens_limit == 4097

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert caplog.records[0].message == (
                "OpenAI API key is missing. You will need to provide an API key to Pipeline.run()."
            )

    @pytest.mark.unit
    def test_init_with_parameters(self, caplog):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            callback = lambda x: x
            component = ChatGPTBackend(
                api_key="test-api-key",
                model_name="gpt-4",
                model_parameters={"max_tokens": 100, "extra-param": "value"},
                streaming_callback=callback,
                api_base_url="test-base-url",
            )
            assert component.api_key == "test-api-key"
            assert component.model_name == "gpt-4"
            assert component.model_parameters == {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 1,
                "n": 1,
                "stop": [],
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "logit_bias": {},
                "stream": False,
                "openai_organization": None,
                "extra-param": "value",
            }
            assert component.streaming_callback == callback
            assert component.api_base_url == "test-base-url"
            assert component.max_tokens_limit == 8192

            tiktoken_patch.get_encoding.assert_called_once_with("cl100k_base")
            assert not caplog.records

    @pytest.mark.unit
    def test_init_unknown_tokenizer(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            with pytest.raises(ValueError, match="Tokenizer for model 'test-another-model-name' not found."):
                ChatGPTBackend(model_name="test-another-model-name")

    @pytest.mark.unit
    def test_init_unknown_token_limit(self, monkeypatch):
        monkeypatch.setattr(
            "haystack.preview.llm_backends.openai.chatgpt.OPENAI_TOKENIZERS", {"test-model-name": "test-encoding"}
        )
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            with pytest.raises(ValueError, match="Max tokens limit for model 'test-model-name' not found."):
                ChatGPTBackend(model_name="test-model-name")

    @pytest.mark.unit
    def test_run_no_api_key(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            component = ChatGPTBackend()
            with pytest.raises(ValueError, match="OpenAI API key is missing. Please provide an API key."):
                component.complete(chat=[])

    @pytest.mark.unit
    def test_complete(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            with patch("haystack.preview.llm_backends.openai.chatgpt.complete") as complete_patch:
                complete_patch.side_effect = lambda payload, **kwargs: (
                    [
                        f"Response for {payload['messages'][1]['content']}",
                        f"Another Response for {payload['messages'][1]['content']}",
                    ],
                    [{"metadata of": payload["messages"][1]["content"]}],
                )
                component = ChatGPTBackend(
                    api_key="test-api-key",
                    model_parameters={"openai_organization": "test_orga_id"},
                    api_base_url="test-base-url",
                )

                results = component.complete(
                    chat=[
                        ChatMessage(content="test-prompt-system", role="system"),
                        ChatMessage(content="test-prompt-user", role="user"),
                    ]
                )

                assert results == (
                    [f"Response for test-prompt-user", f"Another Response for test-prompt-user"],
                    [{"metadata of": "test-prompt-user"}],
                )

                complete_patch.call_count == 2
                complete_patch.assert_called_once_with(
                    url="test-base-url/chat/completions",
                    headers={
                        "Authorization": f"Bearer test-api-key",
                        "Content-Type": "application/json",
                        "OpenAI-Organization": "test_orga_id",
                    },
                    payload={
                        "model": "gpt-3.5-turbo",
                        "max_tokens": 500,
                        "temperature": 0.7,
                        "top_p": 1,
                        "n": 1,
                        "stream": False,
                        "stop": [],
                        "presence_penalty": 0,
                        "frequency_penalty": 0,
                        "logit_bias": {},
                        "messages": [
                            {"role": "system", "content": "test-prompt-system"},
                            {"role": "user", "content": "test-prompt-user"},
                        ],
                    },
                )

    @pytest.mark.unit
    def test_complete_streaming(self):
        with patch("haystack.preview.llm_backends.openai.chatgpt.tiktoken") as tiktoken_patch:
            with patch("haystack.preview.llm_backends.openai.chatgpt.complete_stream") as complete_stream_patch:
                complete_stream_patch.side_effect = lambda payload, **kwargs: (
                    [f"Response for {payload['messages'][1]['content']}"],
                    [{"metadata of": payload["messages"][1]["content"]}],
                )
                callback = Mock()
                component = ChatGPTBackend(api_key="test-api-key", streaming_callback=callback)

                results = component.complete(
                    chat=[
                        ChatMessage(content="test-prompt-system", role="system"),
                        ChatMessage(content="test-prompt-user", role="user"),
                    ]
                )

                assert results == (["Response for test-prompt-user"], [{"metadata of": "test-prompt-user"}])
                complete_stream_patch.call_count == 2
                complete_stream_patch.assert_any_call(
                    url="https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer test-api-key", "Content-Type": "application/json"},
                    payload={
                        "model": "gpt-3.5-turbo",
                        "max_tokens": 500,
                        "temperature": 0.7,
                        "top_p": 1,
                        "n": 1,
                        "stream": True,
                        "stop": [],
                        "presence_penalty": 0,
                        "frequency_penalty": 0,
                        "logit_bias": {},
                        "messages": [
                            {"role": "system", "content": "test-prompt-system"},
                            {"role": "user", "content": "test-prompt-user"},
                        ],
                    },
                    callback=callback,
                )
