# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from openai import OpenAIError

from haystack import Pipeline
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret


class TestOpenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        with pytest.raises(OpenAIError):
            AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")

    def test_init_with_parameters(self):
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-35-turbo"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-35-turbo",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": 30.0,
                "max_retries": 5,
                "default_headers": {},
                "azure_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            azure_ad_token=Secret.from_env_var("ENV_VAR1", strict=False),
            azure_endpoint="some-non-existing-endpoint",
            timeout=2.5,
            max_retries=10,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["ENV_VAR1"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-35-turbo",
                "organization": None,
                "streaming_callback": None,
                "timeout": 2.5,
                "max_retries": 10,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "default_headers": {},
                "azure_kwargs": {},
            },
        }

    def test_pipeline_serialization_deserialization(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        p = Pipeline()
        p.add_component(instance=generator, name="generator")
        p_str = p.dumps()
        q = Pipeline.loads(p_str)
        assert p.to_dict() == q.to_dict(), "Pipeline serialization/deserialization w/ AzureOpenAIChatGenerator failed."

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
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AzureOpenAIChatGenerator(organization="HaystackCI")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.content
        assert "gpt-35-turbo" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    # additional tests intentionally omitted as they are covered by test_openai.py
