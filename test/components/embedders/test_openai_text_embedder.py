# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import CreateEmbeddingResponse, Embedding

import haystack.components.embedders.openai_text_embedder as openai_text_embedder_module
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.utils.auth import Secret


class TestOpenAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAITextEmbedder()

        assert embedder.api_key.resolve_value() == "fake-api-key"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.api_base_url is None
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.client is None
        assert embedder.async_client is None

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
            timeout=40.0,
            max_retries=1,
        )
        assert embedder.api_key.resolve_value() == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.api_base_url == "https://my-custom-base-url.com"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.timeout == 40.0
        assert embedder.max_retries == 1
        assert embedder.client is None
        assert embedder.async_client is None

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_key.resolve_value() == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.api_base_url == "https://my-custom-base-url.com"
        assert embedder.organization == "fake-organization"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.timeout is None
        assert embedder.max_retries is None
        assert embedder.client is None
        assert embedder.async_client is None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        component = OpenAITextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": None,
                "dimensions": None,
                "model": "text-embedding-ada-002",
                "organization": None,
                "http_client_kwargs": None,
                "prefix": "",
                "suffix": "",
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = OpenAITextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            api_base_url="https://my-custom-base-url.com",
            organization="fake-organization",
            prefix="prefix",
            suffix="suffix",
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "api_base_url": "https://my-custom-base-url.com",
                "model": "model",
                "dimensions": None,
                "organization": "fake-organization",
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
                "prefix": "prefix",
                "suffix": "suffix",
                "timeout": 10.0,
                "max_retries": 2,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "text-embedding-ada-002",
                "api_base_url": "https://my-custom-base-url.com",
                "organization": "fake-organization",
                "http_client_kwargs": None,
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }
        component = OpenAITextEmbedder.from_dict(data)
        assert component.api_key.resolve_value() == "fake-api-key"
        assert component.model == "text-embedding-ada-002"
        assert component.api_base_url == "https://my-custom-base-url.com"
        assert component.organization == "fake-organization"
        assert component.http_client_kwargs is None
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"

    def test_prepare_input(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAITextEmbedder(dimensions=1536)

        inp = "The food was delicious"
        prepared_input = embedder._prepare_input(inp)
        assert prepared_input == {
            "model": "text-embedding-ada-002",
            "input": "The food was delicious",
            "encoding_format": "float",
            "dimensions": 1536,
        }

    def test_prepare_output(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")

        response = CreateEmbeddingResponse(
            data=[Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")],
            model="text-embedding-ada-002",
            object="list",
            usage={"prompt_tokens": 6, "total_tokens": 6},
        )

        embedder = OpenAITextEmbedder()
        result = embedder._prepare_output(result=response)
        assert result == {
            "embedding": [0.1, 0.2, 0.3],
            "meta": {"model": "text-embedding-ada-002", "usage": {"prompt_tokens": 6, "total_tokens": 6}},
        }

    def test_run_wrong_input_format(self):
        embedder = OpenAITextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAITextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "text-embedding-ada-002"

        embedder = OpenAITextEmbedder(model=model, prefix="prefix ", suffix=" suffix")
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1536
        assert all(isinstance(x, float) for x in result["embedding"])

        assert "text" in result["meta"]["model"] and "ada" in result["meta"]["model"], (
            "The model name does not contain 'text' and 'ada'"
        )

        assert result["meta"]["usage"] == {"prompt_tokens": 6, "total_tokens": 6}, "Usage information does not match"

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = OpenAITextEmbedder(model="text-embedding-ada-002", prefix="prefix ", suffix=" suffix")
        result = await embedder.run_async(text="The food was delicious")

        assert len(result["embedding"]) == 1536
        assert all(isinstance(x, float) for x in result["embedding"])

        assert "text" in result["meta"]["model"] and "ada" in result["meta"]["model"], (
            "The model name does not contain 'text' and 'ada'"
        )

        assert result["meta"]["usage"] == {"prompt_tokens": 6, "total_tokens": 6}, "Usage information does not match"

        # Close async client; suppress RuntimeError if the event loop is already closed
        with contextlib.suppress(RuntimeError):
            await embedder.close_async()


@pytest.fixture
def mock_openai_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    sync_cls = MagicMock(name="OpenAI")
    async_cls = MagicMock(name="AsyncOpenAI")
    async_cls.return_value.close = AsyncMock()
    monkeypatch.setattr(openai_text_embedder_module, "OpenAI", sync_cls)
    monkeypatch.setattr(openai_text_embedder_module, "AsyncOpenAI", async_cls)
    return sync_cls, async_cls


class TestComponentLifecycle:
    def test_warm_up_uses_default_timeout_and_max_retries(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAITextEmbedder()
        embedder.warm_up()
        assert embedder.client.max_retries == 5
        assert embedder.client.timeout == 30.0

    def test_warm_up_uses_timeout_and_max_retries_from_parameters(self):
        embedder = OpenAITextEmbedder(api_key=Secret.from_token("fake-api-key"), timeout=40.0, max_retries=1)
        embedder.warm_up()
        assert embedder.client.max_retries == 1
        assert embedder.client.timeout == 40.0

    def test_warm_up_uses_timeout_and_max_retries_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAITextEmbedder(api_key=Secret.from_token("fake-api-key"))
        embedder.warm_up()
        assert embedder.client.max_retries == 10
        assert embedder.client.timeout == 100.0

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        embedder = OpenAITextEmbedder()
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            embedder.warm_up()

    def test_sync_lifecycle(self, mock_openai_clients):
        sync_cls, _ = mock_openai_clients
        embedder = OpenAITextEmbedder()
        assert embedder.client is None
        assert embedder.async_client is None

        embedder.warm_up()
        assert embedder.client is sync_cls.return_value
        assert embedder.async_client is None

        embedder.close()
        sync_cls.return_value.close.assert_called_once()
        assert embedder.client is None

    async def test_async_lifecycle(self, mock_openai_clients):
        _, async_cls = mock_openai_clients
        embedder = OpenAITextEmbedder()

        await embedder.warm_up_async()
        assert embedder.async_client is async_cls.return_value
        assert embedder.client is None

        await embedder.close_async()
        async_cls.return_value.close.assert_awaited_once()
        assert embedder.async_client is None

    async def test_close_is_safe_without_warm_up(self, mock_openai_clients):
        embedder = OpenAITextEmbedder()
        embedder.close()
        await embedder.close_async()
        assert embedder.client is None
        assert embedder.async_client is None

    async def test_close_and_close_async_are_independent(self, mock_openai_clients):
        embedder = OpenAITextEmbedder()
        embedder.warm_up()
        await embedder.warm_up_async()

        embedder.close()
        assert embedder.client is None
        assert embedder.async_client is not None

        await embedder.close_async()
        assert embedder.async_client is None
