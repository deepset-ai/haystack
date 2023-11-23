import os

import pytest
import cohere

from haystack.preview.components.generators import CohereGenerator


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from Cohere API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    print(chunk.text, flush=True, end="")


class TestGPTGenerator:
    def test_init_default(self):
        component = CohereGenerator(api_key="test-api-key")
        assert component.api_key == "test-api-key"
        assert component.model == "command"
        assert component.streaming_callback is None
        assert component.api_base_url == cohere.COHERE_API_URL
        assert component.model_parameters == {}

    def test_init_with_parameters(self):
        callback = lambda x: x
        component = CohereGenerator(
            api_key="test-api-key",
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=callback,
            api_base_url="test-base-url",
        )
        assert component.api_key == "test-api-key"
        assert component.model == "command-light"
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self):
        component = CohereGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.cohere.CohereGenerator",
            "init_parameters": {
                "model": "command",
                "streaming_callback": None,
                "api_base_url": cohere.COHERE_API_URL,
            },
        }

    def test_to_dict_with_parameters(self):
        component = CohereGenerator(
            api_key="test-api-key",
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.cohere.CohereGenerator",
            "init_parameters": {
                "model": "command-light",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "test_cohere_generators.default_streaming_callback",
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self):
        component = CohereGenerator(
            api_key="test-api-key",
            model="command",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.preview.components.generators.cohere.CohereGenerator",
            "init_parameters": {
                "model": "command",
                "streaming_callback": "test_cohere_generators.<lambda>",
                "api_base_url": "test-base-url",
                "max_tokens": 10,
                "some_test_param": "test-params",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        data = {
            "type": "haystack.preview.components.generators.cohere.CohereGenerator",
            "init_parameters": {
                "model": "command",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "test_cohere_generators.default_streaming_callback",
            },
        }
        component = CohereGenerator.from_dict(data)
        assert component.api_key = "test-key"
        assert component.model == "command"
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_check_truncated_answers(self, caplog):
        component = CohereGenerator(api_key="test-api-key")
        metadata = [{"finish_reason": "MAX_TOKENS"}]
        component._check_truncated_answers(metadata)
        assert caplog.records[0].message == (
            "Responses have been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run(self):
        component = CohereGenerator(api_key=os.environ.get("COHERE_API_KEY"))
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["metadata"]) == 1
        assert results["metadata"][0]["finish_reason"] == "COMPLETE"

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_wrong_model_name(self):
        component = CohereGenerator(model="something-obviously-wrong", api_key=os.environ.get("COHERE_API_KEY"))
        with pytest.raises(
            cohere.CohereAPIError,
            match="model not found, make sure the correct model ID was used and that you have access to the model.",
        ):
            component.run(prompt="What's the capital of France?")

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""

            def __call__(self, chunk):
                self.responses += chunk.text
                return chunk

        callback = Callback()
        component = CohereGenerator(os.environ.get("COHERE_API_KEY"), streaming_callback=callback)
        results = component.run(prompt="What's the capital of France?")

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["metadata"]) == 1
        assert results["metadata"][0]["finish_reason"] == "COMPLETE"
        assert callback.responses == results["replies"][0]
