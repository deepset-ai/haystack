import os
import logging
from unittest.mock import MagicMock, patch
import pytest
from haystack.utils import Secret
from haystack.components.generators.watsonx import WatsonxGenerator
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from haystack.dataclasses import StreamingChunk

logger = logging.getLogger(__name__)


class TestWatsonxGenerator:
    @pytest.fixture
    def mock_watsonx(self, monkeypatch):
        """Fixture for setting up common mocks"""
        monkeypatch.setenv("WATSONX_API_KEY", "fake-api-key")

        with patch("haystack.components.generators.watsonx.ModelInference") as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance

            # Configure mock responses
            mock_model_instance.generate_text.return_value = {
                "results": [{"generated_text": "This is a generated response"}],
                "model_id": "ibm/granite-13b-instruct-v2",
                "stop_reason": "completed",
            }

            mock_model_instance.generate_text_stream.return_value = [
                {"results": [{"generated_text": "Streaming"}], "stop_reason": None},
                {"results": [{"generated_text": " response"}], "stop_reason": "completed"},
            ]

            yield {"model": mock_model, "model_instance": mock_model_instance}

    def test_init_default(self, mock_watsonx):
        generator = WatsonxGenerator(model="ibm/granite-13b-instruct-v2", project_id="fake-project-id")

        args, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-13b-instruct-v2"
        assert kwargs["project_id"] == "fake-project-id"
        assert kwargs["space_id"] is None
        assert kwargs["params"] == {}
        assert kwargs["verify"] is None

        assert generator.model == "ibm/granite-13b-instruct-v2"
        assert generator.project_id == "fake-project-id"
        assert generator.space_id is None
        assert generator.api_base_url is None
        assert generator.generation_kwargs == {}

    def test_init_with_all_params(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-13b-instruct-v2",
            project_id="test-project",
            space_id="test-space",
            api_base_url="https://custom-url.com",
            generation_kwargs={
                GenParams.MAX_NEW_TOKENS: 100,
                GenParams.TEMPERATURE: 0.7,
                GenParams.DECODING_METHOD: "sample",
                GenParams.TOP_P: 0.9,
            },
            verify=False,
        )

        args, kwargs = mock_watsonx["model"].call_args
        assert kwargs["model_id"] == "ibm/granite-13b-instruct-v2"
        assert kwargs["project_id"] == "test-project"
        assert kwargs["space_id"] == "test-space"
        assert kwargs["params"] == {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "decoding_method": "sample",
            "top_p": 0.9,
        }
        assert kwargs["verify"] is False

    def test_init_fails_without_project_or_space(self, mock_watsonx):
        with pytest.raises(ValueError, match="Either project_id or space_id must be provided"):
            WatsonxGenerator(api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2")

    def test_to_dict(self, mock_watsonx):
        generator = WatsonxGenerator(
            model="ibm/granite-13b-instruct-v2", project_id="test-project", generation_kwargs={"max_new_tokens": 100}
        )

        data = generator.to_dict()

        assert data == {
            "type": "haystack.components.generators.watsonx.WatsonxGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-13b-instruct-v2",
                "project_id": "test-project",
                "space_id": None,
                "api_base_url": None,
                "generation_kwargs": {"max_new_tokens": 100},
                "verify": None,
            },
        }

    def test_from_dict(self, mock_watsonx):
        data = {
            "type": "haystack.components.generators.watsonx.WatsonxGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/granite-13b-instruct-v2",
                "project_id": "test-project",
                "generation_kwargs": {"max_new_tokens": 100},
            },
        }

        generator = WatsonxGenerator.from_dict(data)

        assert generator.model == "ibm/granite-13b-instruct-v2"
        assert generator.project_id == "test-project"
        assert generator.generation_kwargs == {"max_new_tokens": 100}

    def test_run_single_prompt(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("Test prompt")

        assert result["replies"] == ["This is a generated response"]
        assert len(result["meta"]) == 1
        assert result["meta"][0]["model"] == "ibm/granite-13b-instruct-v2"
        assert result["meta"][0]["finish_reason"] == "completed"

        mock_watsonx["model_instance"].generate_text.assert_called_once_with(
            prompt="Test prompt", params={}, guardrails=False, raw_response=True
        )

    def test_run_with_watsonx_params(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="ibm/granite-13b-instruct-v2",
            project_id="test-project",
            generation_kwargs={
                GenParams.MAX_NEW_TOKENS: 100,
                GenParams.TEMPERATURE: 0.7,
                GenParams.DECODING_METHOD: "sample",
                GenParams.TOP_P: 0.9,
            },
        )

        result = generator.run("Test prompt")

        assert result["replies"] == ["This is a generated response"]
        mock_watsonx["model_instance"].generate_text.assert_called_once_with(
            prompt="Test prompt",
            params={"max_new_tokens": 100, "temperature": 0.7, "decoding_method": "sample", "top_p": 0.9},
            guardrails=False,
            raw_response=True,
        )

    def test_run_with_guardrails(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("Test prompt", guardrails=True)

        assert result["replies"] == ["This is a generated response"]
        mock_watsonx["model_instance"].generate_text.assert_called_once_with(
            prompt="Test prompt", params={}, guardrails=True, raw_response=True
        )

    def test_run_with_streaming(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("Test prompt", stream=True)

        assert result["replies"] == ["Streaming response"]
        assert len(result["chunks"]) == 2
        assert isinstance(result["chunks"][0], StreamingChunk)
        assert result["chunks"][0].content == "Streaming"
        assert result["meta"][0]["chunk_count"] == 2

        mock_watsonx["model_instance"].generate_text_stream.assert_called_once_with(
            prompt="Test prompt", params={}, guardrails=False, raw_response=True
        )

    def test_run_with_empty_prompt(self, mock_watsonx):
        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("")
        assert result["replies"] == [""]
        assert result["meta"][0]["finish_reason"] == "empty_input"

    def test_run_with_empty_response(self, mock_watsonx):
        mock_watsonx["model_instance"].generate_text.return_value = {"results": [{}]}

        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("Test prompt")
        assert result["replies"] == ["[Empty response]"]
        assert result["meta"][0]["finish_reason"] == "completed"

    def test_run_with_api_error(self, mock_watsonx):
        mock_watsonx["model_instance"].generate_text.side_effect = Exception("API error")

        generator = WatsonxGenerator(
            api_key=Secret.from_token("test-api-key"), model="ibm/granite-13b-instruct-v2", project_id="test-project"
        )

        result = generator.run("Test prompt")
        assert result["replies"] == ["[Error: API error]"]
        assert result["meta"][0]["finish_reason"] == "error"
        assert "error" in result["meta"][0]


@pytest.mark.integration
class TestWatsonxGeneratorIntegration:
    """Integration tests for WatsonxGenerator (requires real credentials)"""

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_run(self):
        generator = WatsonxGenerator(
            model="ibm/granite-13b-instruct-v2",
            project_id=os.environ["WATSONX_PROJECT_ID"],
            generation_kwargs={"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9},
        )
        results = generator.run("What's the capital of France?")

        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], str)

        if "meta" in results:
            assert isinstance(results["meta"], list)
            assert len(results["meta"]) == 1
            assert results["meta"][0]["model"] == "ibm/granite-13b-instruct-v2"

    @pytest.mark.skipif(
        not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
        reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
    )
    def test_live_streaming(self):
        generator = WatsonxGenerator(model="ibm/granite-13b-instruct-v2", project_id=os.environ["WATSONX_PROJECT_ID"])

        results = generator.run("Explain quantum computing", stream=True)

        assert isinstance(results, dict)
        assert "replies" in results
        assert "chunks" in results
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], str)
        assert all(isinstance(chunk, StreamingChunk) for chunk in results["chunks"])
