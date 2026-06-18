# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai import AsyncOpenAI
from openai.types import ImagesResponse
from openai.types.image import Image

import haystack.components.generators.openai_image_generator as openai_image_generator_module
from haystack.components.generators.openai_image_generator import OpenAIImageGenerator
from haystack.utils import Secret


@pytest.fixture
def mock_image_response():
    with patch("openai.resources.images.Images.generate") as mock_image_generate:
        image_response = ImagesResponse(
            created=1630000000, data=[Image(b64_json="test-b64-json", revised_prompt="test-prompt")]
        )
        mock_image_generate.return_value = image_response
        yield mock_image_generate


class TestOpenAIImageGenerator:
    def test_init_default(self, monkeypatch):
        component = OpenAIImageGenerator()
        assert component.model == "gpt-image-2"
        assert component.quality == "auto"
        assert component.size == "1024x1024"
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.api_base_url is None
        assert component.organization is None
        assert component.timeout is None
        assert component.max_retries is None
        assert component.http_client_kwargs is None
        assert component.client is None
        assert component.async_client is None

    def test_init_with_params(self, monkeypatch):
        component = OpenAIImageGenerator(
            model="gpt-image-1",
            quality="high",
            size="1024x1536",
            api_key=Secret.from_env_var("EXAMPLE_API_KEY"),
            api_base_url="https://api.openai.com",
            organization="test-org",
            timeout=60,
            max_retries=10,
        )
        assert component.model == "gpt-image-1"
        assert component.quality == "high"
        assert component.size == "1024x1536"
        assert component.api_key == Secret.from_env_var("EXAMPLE_API_KEY")
        assert component.api_base_url == "https://api.openai.com"
        assert component.organization == "test-org"
        assert pytest.approx(component.timeout) == 60.0
        assert component.max_retries == 10
        assert component.client is None
        assert component.async_client is None

    def test_init_max_retries_0(self, monkeypatch):
        component = OpenAIImageGenerator(max_retries=0)
        assert component.max_retries == 0

    def test_init_invalid_quality_falls_back_to_auto(self, caplog):
        component = OpenAIImageGenerator(quality="hd")  # type: ignore[arg-type]
        assert component.quality == "auto"
        assert "Invalid quality" in caplog.text

    def test_init_non_default_response_format_warns(self, caplog):
        OpenAIImageGenerator(response_format="url")  # type: ignore[arg-type]
        assert "response_format is ignored" in caplog.text

    def test_to_dict(self):
        generator = OpenAIImageGenerator()
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.openai_image_generator.OpenAIImageGenerator",
            "init_parameters": {
                "model": "gpt-image-2",
                "quality": "auto",
                "size": "1024x1024",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_params(self):
        generator = OpenAIImageGenerator(
            model="gpt-image-1",
            quality="high",
            size="1024x1536",
            api_key=Secret.from_env_var("EXAMPLE_API_KEY"),
            api_base_url="https://api.openai.com",
            organization="test-org",
            timeout=60,
            max_retries=10,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = generator.to_dict()
        assert data == {
            "type": "haystack.components.generators.openai_image_generator.OpenAIImageGenerator",
            "init_parameters": {
                "model": "gpt-image-1",
                "quality": "high",
                "size": "1024x1536",
                "api_key": {"type": "env_var", "env_vars": ["EXAMPLE_API_KEY"], "strict": True},
                "api_base_url": "https://api.openai.com",
                "organization": "test-org",
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.generators.openai_image_generator.OpenAIImageGenerator",
            "init_parameters": {
                "model": "gpt-image-2",
                "quality": "auto",
                "size": "1024x1024",
                "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                "api_base_url": None,
                "organization": None,
                "http_client_kwargs": None,
            },
        }
        generator = OpenAIImageGenerator.from_dict(data)
        assert generator.model == "gpt-image-2"
        assert generator.quality == "auto"
        assert generator.size == "1024x1024"
        assert generator.api_key.to_dict() == {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True}
        assert generator.http_client_kwargs is None

    def test_from_dict_default_params(self):
        data = {
            "type": "haystack.components.generators.openai_image_generator.OpenAIImageGenerator",
            "init_parameters": {},
        }
        generator = OpenAIImageGenerator.from_dict(data)
        assert generator.model == "gpt-image-2"
        assert generator.quality == "auto"
        assert generator.size == "1024x1024"
        assert generator.api_key.to_dict() == {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True}
        assert generator.api_base_url is None
        assert generator.organization is None
        assert generator.timeout is None
        assert generator.max_retries is None
        assert generator.http_client_kwargs is None

    def test_run(self, mock_image_response):
        generator = OpenAIImageGenerator(api_key=Secret.from_token("test-api-key"))
        response = generator.run("Show me a picture of a black cat.")
        assert generator.client is not None
        assert isinstance(response, dict)
        assert "images" in response and "revised_prompt" in response
        assert response["images"] == ["test-b64-json"]
        assert response["revised_prompt"] == "test-prompt"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.slow
    def test_live_run(self):
        generator = OpenAIImageGenerator(model="gpt-image-1-mini", size="1024x1024", quality="low")
        response = generator.run("A nice cat")
        assert isinstance(response, dict)
        assert isinstance(response["revised_prompt"], str)

        image_str = response["images"][0]
        assert isinstance(image_str, str) and image_str

        decoded = base64.b64decode(image_str, validate=True)
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


class TestOpenAIImageGeneratorAsync:
    def test_async_client_none_before_warm_up(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIImageGenerator()
        assert component.async_client is None

    @pytest.mark.asyncio
    async def test_async_client_after_warm_up_async(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIImageGenerator()
        await component.warm_up_async()
        assert isinstance(component.async_client, AsyncOpenAI)
        assert component.async_client.api_key == "test-api-key"

    @pytest.mark.asyncio
    async def test_run_async(self):
        generator = OpenAIImageGenerator(api_key=Secret.from_token("test-api-key"))

        image_response = ImagesResponse(
            created=1630000000, data=[Image(b64_json="test-b64-json", revised_prompt="test-prompt")]
        )
        mock_async_client = Mock()
        mock_async_client.images.generate = AsyncMock(return_value=image_response)
        generator.async_client = mock_async_client

        response = await generator.run_async("Show me a picture of a black cat.")
        assert isinstance(response, dict)
        assert "images" in response and "revised_prompt" in response
        assert response["images"] == ["test-b64-json"]
        assert response["revised_prompt"] == "test-prompt"
        mock_async_client.images.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_async_triggers_warm_up(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        generator = OpenAIImageGenerator()
        assert generator.async_client is None

        image_response = ImagesResponse(
            created=1630000000, data=[Image(b64_json="test-b64-json", revised_prompt="test-prompt")]
        )

        with patch("openai.resources.images.AsyncImages.generate", new=AsyncMock(return_value=image_response)):
            response = await generator.run_async("Show me a picture of a black cat.")

        assert isinstance(generator.async_client, AsyncOpenAI)
        assert response["images"] == ["test-b64-json"]
        assert response["revised_prompt"] == "test-prompt"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_live_run_async(self):
        generator = OpenAIImageGenerator(model="gpt-image-1-mini", size="1024x1024", quality="low")
        response = await generator.run_async("A nice cat")
        assert isinstance(response, dict)
        assert isinstance(response["revised_prompt"], str)

        image_str = response["images"][0]
        assert isinstance(image_str, str) and image_str

        decoded = base64.b64decode(image_str, validate=True)
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def mock_openai_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    sync_cls = MagicMock(name="OpenAI")
    async_cls = MagicMock(name="AsyncOpenAI")
    async_cls.return_value.close = AsyncMock()
    monkeypatch.setattr(openai_image_generator_module, "OpenAI", sync_cls)
    monkeypatch.setattr(openai_image_generator_module, "AsyncOpenAI", async_cls)
    return sync_cls, async_cls


class TestComponentLifecycle:
    def test_warm_up_uses_default_timeout_and_max_retries(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        generator = OpenAIImageGenerator()
        generator.warm_up()
        assert generator.client.max_retries == 5
        assert generator.client.timeout == 30.0

    def test_warm_up_uses_timeout_and_max_retries_from_parameters(self):
        generator = OpenAIImageGenerator(api_key=Secret.from_token("fake-api-key"), timeout=40.0, max_retries=1)
        generator.warm_up()
        assert generator.client.max_retries == 1
        assert generator.client.timeout == 40.0

    def test_warm_up_uses_timeout_and_max_retries_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        generator = OpenAIImageGenerator(api_key=Secret.from_token("fake-api-key"))
        generator.warm_up()
        assert generator.client.max_retries == 10
        assert generator.client.timeout == 100.0

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        generator = OpenAIImageGenerator()
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            generator.warm_up()

    def test_sync_lifecycle(self, mock_openai_clients):
        sync_cls, _ = mock_openai_clients
        generator = OpenAIImageGenerator()
        assert generator.client is None
        assert generator.async_client is None

        generator.warm_up()
        assert generator.client is sync_cls.return_value
        assert generator.async_client is None

        generator.close()
        sync_cls.return_value.close.assert_called_once()
        assert generator.client is None

    async def test_async_lifecycle(self, mock_openai_clients):
        _, async_cls = mock_openai_clients
        generator = OpenAIImageGenerator()

        await generator.warm_up_async()
        assert generator.async_client is async_cls.return_value
        assert generator.client is None

        await generator.close_async()
        async_cls.return_value.close.assert_awaited_once()
        assert generator.async_client is None

    async def test_close_is_safe_without_warm_up(self, mock_openai_clients):
        generator = OpenAIImageGenerator()
        generator.close()
        await generator.close_async()
        assert generator.client is None
        assert generator.async_client is None

    async def test_close_and_close_async_are_independent(self, mock_openai_clients):
        generator = OpenAIImageGenerator()
        generator.warm_up()
        await generator.warm_up_async()

        generator.close()
        assert generator.client is None
        assert generator.async_client is not None

        await generator.close_async()
        assert generator.async_client is None
