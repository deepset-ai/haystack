import os
from unittest.mock import patch, Mock
from copy import deepcopy

import pytest
import openai
from openai.util import convert_to_openai_object

from haystack.preview.components.generators.openai.gpt import GPTGenerator
from haystack.preview.components.generators.openai.gpt import default_streaming_callback


def mock_openai_response(messages: str, model: str = "gpt-3.5-turbo-0301", **kwargs) -> openai.ChatCompletion:
    response = f"response for these messages --> {' - '.join(msg['role']+': '+msg['content'] for msg in messages)}"
    base_dict = {
        "id": "chatcmpl-7NaPEA6sgX7LnNPyKPbRlsyqLbr5V",
        "object": "chat.completion",
        "created": 1685855844,
        "model": model,
        "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
    }
    base_dict["choices"] = [
        {"message": {"role": "assistant", "content": response}, "finish_reason": "stop", "index": "0"}
    ]
    return convert_to_openai_object(deepcopy(base_dict))


def mock_openai_stream_response(messages: str, model: str = "gpt-3.5-turbo-0301", **kwargs) -> openai.ChatCompletion:
    response = f"response for these messages --> {' - '.join(msg['role']+': '+msg['content'] for msg in messages)}"
    base_dict = {
        "id": "chatcmpl-7NaPEA6sgX7LnNPyKPbRlsyqLbr5V",
        "object": "chat.completion",
        "created": 1685855844,
        "model": model,
    }
    base_dict["choices"] = [{"delta": {"role": "assistant"}, "finish_reason": None, "index": "0"}]
    yield convert_to_openai_object(base_dict)
    for token in response.split():
        base_dict["choices"][0]["delta"] = {"content": token + " "}
        yield convert_to_openai_object(base_dict)
    base_dict["choices"] = [{"delta": {"content": ""}, "finish_reason": "stop", "index": "0"}]
    yield convert_to_openai_object(base_dict)


class TestGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self):
        component = GPTGenerator(api_key="test-api-key")
        assert component.system_prompt is None
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-3.5-turbo"
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.openai.com/v1"
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        callback = lambda x: x
        component = GPTGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            system_prompt="test-system-prompt",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=callback,
            api_base_url="test-base-url",
        )
        assert component.system_prompt == "test-system-prompt"
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4"
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_to_dict_default(self):
        component = GPTGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "GPTGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-3.5-turbo",
                "system_prompt": None,
                "streaming_callback": None,
                "api_base_url": "https://api.openai.com/v1",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = GPTGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            system_prompt="test-system-prompt",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "GPTGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4",
                "system_prompt": "test-system-prompt",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.openai.gpt.default_streaming_callback",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_lambda_streaming_callback(self):
        component = GPTGenerator(
            api_key="test-api-key",
            model_name="gpt-4",
            system_prompt="test-system-prompt",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "GPTGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4",
                "system_prompt": "test-system-prompt",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "test_gpt_generator.<lambda>",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "GPTGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model_name": "gpt-4",
                "system_prompt": "test-system-prompt",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.openai.gpt.default_streaming_callback",
            },
        }
        component = GPTGenerator.from_dict(data)
        assert component.system_prompt == "test-system-prompt"
        assert component.api_key == "test-api-key"
        assert component.model_name == "gpt-4"
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_run_no_system_prompt(self):
        with patch("haystack.preview.components.generators.openai.gpt.openai.ChatCompletion") as gpt_patch:
            gpt_patch.create.side_effect = mock_openai_response
            component = GPTGenerator(api_key="test-api-key")
            results = component.run(prompt="test-prompt-1")
            assert results == {
                "replies": ["response for these messages --> user: test-prompt-1"],
                "metadata": [
                    {
                        "model": "gpt-3.5-turbo",
                        "index": "0",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
                    }
                ],
            }
            gpt_patch.create.assert_called_once_with(
                model="gpt-3.5-turbo",
                api_key="test-api-key",
                messages=[{"role": "user", "content": "test-prompt-1"}],
                stream=False,
            )

    @pytest.mark.unit
    def test_run_with_system_prompt(self):
        with patch("haystack.preview.components.generators.openai.gpt.openai.ChatCompletion") as gpt_patch:
            gpt_patch.create.side_effect = mock_openai_response
            component = GPTGenerator(api_key="test-api-key", system_prompt="test-system-prompt")
            results = component.run(prompt="test-prompt-1")
            assert results == {
                "replies": ["response for these messages --> system: test-system-prompt - user: test-prompt-1"],
                "metadata": [
                    {
                        "model": "gpt-3.5-turbo",
                        "index": "0",
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
                    }
                ],
            }
            gpt_patch.create.assert_called_once_with(
                model="gpt-3.5-turbo",
                api_key="test-api-key",
                messages=[
                    {"role": "system", "content": "test-system-prompt"},
                    {"role": "user", "content": "test-prompt-1"},
                ],
                stream=False,
            )

    @pytest.mark.unit
    def test_run_with_parameters(self):
        with patch("haystack.preview.components.generators.openai.gpt.openai.ChatCompletion") as gpt_patch:
            gpt_patch.create.side_effect = mock_openai_response
            component = GPTGenerator(api_key="test-api-key", max_tokens=10)
            component.run(prompt="test-prompt-1")
            gpt_patch.create.assert_called_once_with(
                model="gpt-3.5-turbo",
                api_key="test-api-key",
                messages=[{"role": "user", "content": "test-prompt-1"}],
                stream=False,
                max_tokens=10,
            )

    @pytest.mark.unit
    def test_run_stream(self):
        with patch("haystack.preview.components.generators.openai.gpt.openai.ChatCompletion") as gpt_patch:
            mock_callback = Mock()
            mock_callback.side_effect = default_streaming_callback
            gpt_patch.create.side_effect = mock_openai_stream_response
            component = GPTGenerator(
                api_key="test-api-key", system_prompt="test-system-prompt", streaming_callback=mock_callback
            )
            results = component.run(prompt="test-prompt-1")
            assert results == {
                "replies": ["response for these messages --> system: test-system-prompt - user: test-prompt-1 "],
                "metadata": [{"model": "gpt-3.5-turbo", "index": "0", "finish_reason": "stop"}],
            }
            # Calls count: 10 tokens per prompt + 1 token for the role + 1 empty termination token
            assert mock_callback.call_count == 12
            gpt_patch.create.assert_called_once_with(
                model="gpt-3.5-turbo",
                api_key="test-api-key",
                messages=[
                    {"role": "system", "content": "test-system-prompt"},
                    {"role": "user", "content": "test-prompt-1"},
                ],
                stream=True,
            )

    @pytest.mark.unit
    def test_check_truncated_answers(self, caplog):
        component = GPTGenerator(api_key="test-api-key")
        metadata = [
            {"finish_reason": "stop"},
            {"finish_reason": "content_filter"},
            {"finish_reason": "length"},
            {"finish_reason": "stop"},
        ]
        component._check_truncated_answers(metadata)
        assert caplog.records[0].message == (
            "2 out of the 4 completions have been truncated before reaching a natural "
            "stopping point. Increase the max_tokens parameter to allow for longer completions."
        )

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_gpt_generator_run(self):
        component = GPTGenerator(api_key=os.environ.get("OPENAI_API_KEY"), n=1)
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["metadata"]) == 1
        assert "gpt-3.5" in results["metadata"][0]["model"]
        assert results["metadata"][0]["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_gpt_generator_run_wrong_model_name(self):
        component = GPTGenerator(model_name="something-obviously-wrong", api_key=os.environ.get("OPENAI_API_KEY"), n=1)
        with pytest.raises(openai.InvalidRequestError, match="The model `something-obviously-wrong` does not exist"):
            component.run(prompt="What's the capital of France?")

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_gpt_generator_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""

            def __call__(self, chunk):
                self.responses += chunk.choices[0].delta.content if chunk.choices[0].delta else ""
                return chunk

        callback = Callback()
        component = GPTGenerator(os.environ.get("OPENAI_API_KEY"), streaming_callback=callback, n=1)
        results = component.run(prompt="What's the capital of France?")

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]

        assert len(results["metadata"]) == 1
        assert "gpt-3.5" in results["metadata"][0]["model"]
        assert results["metadata"][0]["finish_reason"] == "stop"

        assert callback.responses == results["replies"][0]
