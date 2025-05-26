# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Optional

import pytest
from openai import OpenAIError

from haystack import component, Pipeline
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import ComponentTool, Tool
from haystack.tools.toolset import Toolset
from haystack.utils.auth import Secret
from haystack.utils.azure import default_azure_ad_token_provider


def get_weather(city: str) -> Dict[str, Any]:
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


@component
class MessageExtractor:
    @component.output_types(messages=List[str], meta=Dict[str, Any])
    def run(self, messages: List[ChatMessage], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts the text content of ChatMessage objects

        :param messages: List of Haystack ChatMessage objects
        :param meta: Optional metadata to include in the response.
        :returns:
            A dictionary with keys "messages" and "meta".
        """
        if meta is None:
            meta = {}
        return {"messages": [m.text for m in messages], "meta": meta}


@pytest.fixture
def tools():
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=get_weather,
    )
    # We add a tool that has a more complex parameter signature
    message_extractor_tool = ComponentTool(
        component=MessageExtractor(),
        name="message_extractor",
        description="Useful for returning the text content of ChatMessage objects",
    )
    return [weather_tool, message_extractor_tool]


class TestAzureOpenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-4o-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        with pytest.raises(OpenAIError):
            AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")

    def test_init_with_parameters(self, tools):
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
            azure_ad_token_provider=default_azure_ad_token_provider,
        )
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict
        assert component.azure_ad_token_provider is not None
        assert component.max_retries == 5

    def test_init_with_0_max_retries(self, tools):
        """Tests that the max_retries init param is set correctly if equal 0"""
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
            azure_ad_token_provider=default_azure_ad_token_provider,
            max_retries=0,
        )
        assert component.client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict
        assert component.azure_ad_token_provider is not None
        assert component.max_retries == 0

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
                "azure_deployment": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": 30.0,
                "max_retries": 5,
                "default_headers": {},
                "tools": None,
                "tools_strict": False,
                "azure_ad_token_provider": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            azure_ad_token=Secret.from_env_var("ENV_VAR1", strict=False),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            timeout=2.5,
            max_retries=10,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            azure_ad_token_provider=default_azure_ad_token_provider,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["ENV_VAR1"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "timeout": 2.5,
                "max_retries": 10,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": None,
                "tools_strict": False,
                "default_headers": {},
                "azure_ad_token_provider": "haystack.utils.azure.default_azure_ad_token_provider",
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "test-ad-token")
        data = {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2023-05-15",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": 30.0,
                "max_retries": 5,
                "default_headers": {},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }

        generator = AzureOpenAIChatGenerator.from_dict(data)
        assert isinstance(generator, AzureOpenAIChatGenerator)

        assert generator.api_key == Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False)
        assert generator.azure_ad_token == Secret.from_env_var("AZURE_OPENAI_AD_TOKEN", strict=False)
        assert generator.api_version == "2023-05-15"
        assert generator.azure_endpoint == "some-non-existing-endpoint"
        assert generator.azure_deployment == "gpt-4o-mini"
        assert generator.organization is None
        assert generator.streaming_callback is None
        assert generator.generation_kwargs == {}
        assert generator.timeout == 30.0
        assert generator.max_retries == 5
        assert generator.default_headers == {}
        assert generator.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert generator.tools_strict == False
        assert generator.http_client_kwargs is None

    def test_pipeline_serialization_deserialization(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        p = Pipeline()
        p.add_component(instance=generator, name="generator")

        assert p.to_dict() == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
                    "init_parameters": {
                        "azure_endpoint": "some-non-existing-endpoint",
                        "azure_deployment": "gpt-4o-mini",
                        "organization": None,
                        "api_version": "2023-05-15",
                        "streaming_callback": None,
                        "generation_kwargs": {},
                        "timeout": 30.0,
                        "max_retries": 5,
                        "api_key": {"type": "env_var", "env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False},
                        "azure_ad_token": {"type": "env_var", "env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False},
                        "default_headers": {},
                        "tools": None,
                        "tools_strict": False,
                        "azure_ad_token_provider": None,
                        "http_client_kwargs": None,
                    },
                }
            },
            "connections": [],
        }
        p_str = p.dumps()
        q = Pipeline.loads(p_str)
        assert p.to_dict() == q.to_dict(), "Pipeline serialization/deserialization w/ AzureOpenAIChatGenerator failed."

    def test_azure_chat_generator_with_toolset_initialization(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be initialized with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
        assert generator.tools == toolset

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be deserialized from a dictionary with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
        data = component.to_dict()

        deserialized_component = AzureOpenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) or not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
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
        assert "Paris" in message.text
        assert "gpt-4o-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) or not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AzureOpenAIChatGenerator(organization="HaystackCI", tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be serialized to a dictionary with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools[:1])
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
        data = component.to_dict()

        expected_tools_data = {
            "type": "haystack.tools.toolset.Toolset",
            "data": {
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "weather",
                            "description": "useful to determine the weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                            "function": "generators.chat.test_azure.get_weather",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    }
                ]
            },
        }
        assert data["init_parameters"]["tools"] == expected_tools_data


class TestAzureOpenAIChatGeneratorAsync:
    def test_init_should_also_create_async_client_with_same_args(self, tools):
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
        )
        assert component.async_client.api_key == "test-api-key"
        assert component.azure_deployment == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) or not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AzureOpenAIChatGenerator(generation_kwargs={"n": 1})
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None) or not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
        reason=(
            "Please export env variables called AZURE_OPENAI_API_KEY containing "
            "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
            "the Azure OpenAI endpoint URL to run this test."
        ),
    )
    @pytest.mark.asyncio
    async def test_live_run_with_tools_async(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AzureOpenAIChatGenerator(tools=tools)
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    # additional tests intentionally omitted as they are covered by test_openai.py
