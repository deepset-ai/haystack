# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch
from haystack.utils import Secret

from openai.types.image import Image
from openai.types import ImagesResponse
from haystack.components.generators.openai_dalle import DALLEImageGenerator


@pytest.fixture
def mock_image_response():
    with patch("openai.resources.images.Images.generate") as mock_image_generate:
        image_response = ImagesResponse(created=1630000000, data=[Image(url="test-url", revised_prompt="test-prompt")])
        mock_image_generate.return_value = image_response
        yield mock_image_generate


class TestDALLEImageGenerator:
    def test_init_default(self, monkeypatch):
        component = DALLEImageGenerator()
        assert component.model == "dall-e-3"
        assert component.quality == "standard"
        assert component.size == "1024x1024"
        assert component.response_format == "url"
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.api_base_url is None
        assert component.organization is None
        assert pytest.approx(component.timeout) == 30.0
        assert component.max_retries is 5

    def test_init_with_params(self, monkeypatch):
        component = DALLEImageGenerator(
            model="dall-e-2",
            quality="hd",
            size="256x256",
            response_format="b64_json",
            api_key=Secret.from_env_var("EXAMPLE_API_KEY"),
            api_base_url="https://api.openai.com",
            organization="test-org",
            timeout=60,
            max_retries=10,
        )
        assert component.model == "dall-e-2"
        assert component.quality == "hd"
        assert component.size == "256x256"
        assert component.response_format == "b64_json"
        assert component.api_key == Secret.from_env_var("EXAMPLE_API_KEY")
        assert component.api_base_url == "https://api.openai.com"
        assert component.organization == "test-org"
        assert pytest.approx(component.timeout) == 60.0
        assert component.max_retries == 10

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DALLEImageGenerator()
        component.warm_up()
        assert component.client.api_key == "test-api-key"
        assert component.client.timeout == 30
        assert component.client.max_retries == 5

    def test_to_dict(self):
        generator = DALLEImageGenerator()
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.openai_dalle.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
            },
        }

    def test_to_dict_with_params(self):
        generator = DALLEImageGenerator(
            model="dall-e-2",
            quality="hd",
            size="256x256",
            response_format="b64_json",
            api_key=Secret.from_env_var("EXAMPLE_API_KEY"),
            api_base_url="https://api.openai.com",
            organization="test-org",
            timeout=60,
            max_retries=10,
        )
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.openai_dalle.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-2",
                "quality": "hd",
                "size": "256x256",
                "response_format": "b64_json",
                "api_key": {"type": "env_var", "env_vars": ["EXAMPLE_API_KEY"], "strict": True},
                "api_base_url": "https://api.openai.com",
                "organization": "test-org",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.generators.openai_dalle.DALLEImageGenerator",
            "init_parameters": {
                "model": "dall-e-3",
                "quality": "standard",
                "size": "1024x1024",
                "response_format": "url",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
            },
        }
        generator = DALLEImageGenerator.from_dict(data)
        assert generator.model == "dall-e-3"
        assert generator.quality == "standard"
        assert generator.size == "1024x1024"
        assert generator.response_format == "url"
        assert generator.api_key.to_dict() == {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True}

    def test_from_dict_default_params(self):
        data = {"type": "haystack.components.generators.openai_dalle.DALLEImageGenerator", "init_parameters": {}}
        generator = DALLEImageGenerator.from_dict(data)
        assert generator.model == "dall-e-3"
        assert generator.quality == "standard"
        assert generator.size == "1024x1024"
        assert generator.response_format == "url"
        assert generator.api_key.to_dict() == {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True}
        assert generator.api_base_url is None
        assert generator.organization is None
        assert pytest.approx(generator.timeout) == 30.0
        assert generator.max_retries == 5

    def test_run(self, mock_image_response):
        generator = DALLEImageGenerator(api_key=Secret.from_token("test-api-key"))
        generator.warm_up()
        response = generator.run("Show me a picture of a black cat.")
        assert isinstance(response, dict)
        assert "images" in response and "revised_prompt" in response
        assert response["images"] == ["test-url"]
        assert response["revised_prompt"] == "test-prompt"
