# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from haystack.components.generators.modelslab import ModelsLabImageGenerator
from haystack.utils import Secret


class TestModelsLabImageGenerator:
    def test_init_default(self):
        generator = ModelsLabImageGenerator()
        assert generator.model == "flux"
        assert generator.width == 512
        assert generator.height == 512
        assert generator.samples == 1
        assert generator.num_inference_steps == 30
        assert pytest.approx(generator.guidance_scale) == 7.5
        assert generator.negative_prompt is None
        assert generator.seed is None
        assert generator.api_key == Secret.from_env_var("MODELSLAB_API_KEY")
        assert generator.api_base_url == "https://modelslab.com/api/v6"

    def test_init_with_params(self):
        generator = ModelsLabImageGenerator(
            model="sdxl",
            width=1024,
            height=1024,
            samples=2,
            num_inference_steps=20,
            guidance_scale=9.0,
            negative_prompt="blurry, ugly",
            seed=42,
            api_key=Secret.from_env_var("MY_API_KEY"),
            api_base_url="https://custom.modelslab.com/api/v6",
        )
        assert generator.model == "sdxl"
        assert generator.width == 1024
        assert generator.height == 1024
        assert generator.samples == 2
        assert generator.num_inference_steps == 20
        assert pytest.approx(generator.guidance_scale) == 9.0
        assert generator.negative_prompt == "blurry, ugly"
        assert generator.seed == 42
        assert generator.api_key == Secret.from_env_var("MY_API_KEY")
        assert generator.api_base_url == "https://custom.modelslab.com/api/v6"

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator()
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.modelslab.ModelsLabImageGenerator",
            "init_parameters": {
                "model": "flux",
                "width": 512,
                "height": 512,
                "samples": 1,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": None,
                "seed": None,
                "api_key": {"type": "env_var", "env_vars": ["MODELSLAB_API_KEY"], "strict": True},
                "api_base_url": "https://modelslab.com/api/v6",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        data = {
            "type": "haystack.components.generators.modelslab.ModelsLabImageGenerator",
            "init_parameters": {
                "model": "sdxl",
                "width": 768,
                "height": 768,
                "samples": 1,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": None,
                "seed": None,
                "api_key": {"type": "env_var", "env_vars": ["MODELSLAB_API_KEY"], "strict": True},
                "api_base_url": "https://modelslab.com/api/v6",
            },
        }
        generator = ModelsLabImageGenerator.from_dict(data)
        assert generator.model == "sdxl"
        assert generator.width == 768
        assert generator.height == 768

    def test_run_success(self, monkeypatch):
        """Test successful synchronous image generation."""
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator(model="flux", width=512, height=512)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "output": ["https://cdn.modelslab.com/image1.png"],
            "generationTime": 2.5,
            "id": 12345,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = generator.run("A beautiful sunset over mountains")

        assert result["images"] == ["https://cdn.modelslab.com/image1.png"]
        assert len(result["metadata"]) == 1
        assert result["metadata"][0]["model"] == "flux"
        assert result["metadata"][0]["width"] == 512
        assert result["metadata"][0]["height"] == 512
        assert pytest.approx(result["metadata"][0]["generation_time"]) == 2.5

        # Verify request payload
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["prompt"] == "A beautiful sunset over mountains"
        assert payload["model_id"] == "flux"
        assert payload["key"] == "test-key"
        assert payload["width"] == 512
        assert payload["height"] == 512

    def test_run_with_overrides(self, monkeypatch):
        """Test run() method respects per-call overrides."""
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator(model="flux", width=512, height=512)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "output": ["https://cdn.modelslab.com/image1.png"],
            "generationTime": 3.1,
            "id": 99999,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            result = generator.run(
                "A futuristic cityscape",
                model="sdxl",
                width=1024,
                height=768,
                negative_prompt="blurry",
            )

        assert result["images"] == ["https://cdn.modelslab.com/image1.png"]
        assert result["metadata"][0]["model"] == "sdxl"
        assert result["metadata"][0]["width"] == 1024

    def test_run_processing_polls(self, monkeypatch):
        """Test that processing status triggers polling until success."""
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator()

        initial_response = MagicMock()
        initial_response.json.return_value = {
            "status": "processing",
            "id": 54321,
            "fetch_result": "https://modelslab.com/api/v6/images/fetch/54321",
            "eta": 5,
        }
        initial_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=initial_response), \
             patch.object(generator, "_poll_for_result", return_value=["https://cdn.modelslab.com/polled.png"]) as mock_poll:
            result = generator.run("A polled image")

        mock_poll.assert_called_once_with(generation_id=54321, api_key="test-key")
        assert result["images"] == ["https://cdn.modelslab.com/polled.png"]

    def test_run_error_raises(self, monkeypatch):
        """Test that an error status raises RuntimeError."""
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "error",
            "message": "Invalid API key",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response), pytest.raises(RuntimeError, match="Invalid API key"):
            generator.run("Test prompt")

    def test_run_processing_missing_id_raises(self, monkeypatch):
        """Test processing status without an ID raises RuntimeError."""
        monkeypatch.setenv("MODELSLAB_API_KEY", "test-key")
        generator = ModelsLabImageGenerator()

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "processing"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response), pytest.raises(RuntimeError, match="generation ID"):
            generator.run("Test prompt")
