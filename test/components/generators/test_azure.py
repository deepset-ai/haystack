# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

from haystack import Pipeline
from haystack.utils.auth import Secret

import pytest
from openai import OpenAIError

from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.generators.utils import print_streaming_chunk


class TestAzureOpenAIGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        with pytest.raises(OpenAIError):
            AzureOpenAIGenerator(azure_endpoint="some-non-existing-endpoint")

    def test_init_with_parameters(self):
        component = AzureOpenAIGenerator(
            api_key=Secret.from_token("fake-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            azure_deployment="gpt-35-turbo",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "fake-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is print_streaming_chunk
        assert component.timeout == 30.0
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIGenerator(azure_endpoint="some-non-existing-endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.azure.AzureOpenAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_deployment": "gpt-35-turbo",
                "api_version": "2023-05-15",
                "streaming_callback": None,
                "azure_endpoint": "some-non-existing-endpoint",
                "organization": None,
                "system_prompt": None,
                "timeout": 30.0,
                "max_retries": 5,
                "generation_kwargs": {},
                "default_headers": {},
                "azure_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AzureOpenAIGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            azure_ad_token=Secret.from_env_var("ENV_VAR1", strict=False),
            azure_endpoint="some-non-existing-endpoint",
            timeout=3.5,
            max_retries=10,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )

        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.azure.AzureOpenAIGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["ENV_VAR1"], "strict": False, "type": "env_var"},
                "azure_deployment": "gpt-35-turbo",
                "api_version": "2023-05-15",
                "streaming_callback": None,
                "azure_endpoint": "some-non-existing-endpoint",
                "organization": None,
                "system_prompt": None,
                "timeout": 3.5,
                "max_retries": 10,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "default_headers": {},
                "azure_kwargs": {},
            },
        }

    def test_pipeline_serialization_deserialization(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        generator = AzureOpenAIGenerator(azure_endpoint="some-non-existing-endpoint")
        p = Pipeline()
        p.add_component(instance=generator, name="generator")
        p_str = p.dumps()
        q = Pipeline.loads(p_str)
        assert p.to_dict() == q.to_dict(), "Pipeline serialization/deserialization with AzureOpenAIGenerator failed."

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) and not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    def test_live_run(self):
        component = AzureOpenAIGenerator(organization="HaystackCI")
        results = component.run("What's the capital of France?")
        assert len(results["replies"]) == 1
        assert len(results["meta"]) == 1
        response: str = results["replies"][0]
        assert "Paris" in response

        metadata = results["meta"][0]
        assert "gpt-35-turbo" in metadata["model"]
        assert metadata["finish_reason"] == "stop"

        assert "usage" in metadata
        assert "prompt_tokens" in metadata["usage"] and metadata["usage"]["prompt_tokens"] > 0
        assert "completion_tokens" in metadata["usage"] and metadata["usage"]["completion_tokens"] > 0
        assert "total_tokens" in metadata["usage"] and metadata["usage"]["total_tokens"] > 0

    # additional tests intentionally omitted as they are covered by test_openai.py
