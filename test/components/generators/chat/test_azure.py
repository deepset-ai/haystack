# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import OpenAIError
from pydantic import BaseModel

import haystack.components.generators.chat.azure as azure_chat_module
from haystack import Pipeline, component
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import ComponentTool, Tool
from haystack.tools.toolset import Toolset
from haystack.utils.auth import Secret
from haystack.utils.azure import default_azure_ad_token_provider


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


@pytest.fixture
def calendar_event_model():
    return CalendarEvent


def get_weather(city: str) -> dict[str, Any]:
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


@component
class MessageExtractor:
    @component.output_types(messages=list[str], meta=dict[str, Any])
    def run(self, messages: list[ChatMessage], meta: dict[str, Any] | None = None) -> dict[str, Any]:
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
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = AzureOpenAIChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.api_key == Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False)
        assert component.azure_deployment == "gpt-4.1-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client is None
        assert component.async_client is None

    def test_init_does_not_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client is None
        assert component.async_client is None

    def test_init_with_parameters(self, tools):
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_completion_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
            azure_ad_token_provider=default_azure_ad_token_provider,
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.azure_deployment == "gpt-4.1-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_completion_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict
        assert component.azure_ad_token_provider is not None
        assert component.max_retries is None
        assert component.client is None
        assert component.async_client is None

    def test_init_with_0_max_retries(self, tools):
        """Tests that the max_retries init param is set correctly if equal 0"""
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_completion_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
            azure_ad_token_provider=default_azure_ad_token_provider,
            max_retries=0,
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.azure_deployment == "gpt-4.1-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_completion_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict
        assert component.azure_ad_token_provider is not None
        assert component.max_retries == 0
        assert component.client is None
        assert component.async_client is None

    def test_init_with_secret_azure_endpoint_and_api_version(self, monkeypatch):
        """`azure_endpoint` and `api_version` accept a Secret that is resolved from an environment variable."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.azure.openai.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        component = AzureOpenAIChatGenerator(
            azure_endpoint=Secret.from_env_var("AZURE_OPENAI_ENDPOINT"),
            api_version=Secret.from_env_var("AZURE_OPENAI_API_VERSION"),
        )
        # The Secret objects are kept on the instance so they can be serialized
        assert component.azure_endpoint == Secret.from_env_var("AZURE_OPENAI_ENDPOINT")
        assert component.api_version == Secret.from_env_var("AZURE_OPENAI_API_VERSION")

    def test_init_fail_with_unset_secret_azure_endpoint(self, monkeypatch):
        """A Secret azure_endpoint that resolves to nothing raises the same error as a missing endpoint."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="Azure endpoint"):
            AzureOpenAIChatGenerator(azure_endpoint=Secret.from_env_var("AZURE_OPENAI_ENDPOINT", strict=False))

    def test_to_dict_with_secret_azure_endpoint_and_api_version(self, monkeypatch):
        """Secret `azure_endpoint` and `api_version` are serialized as Secret dictionaries."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.azure.openai.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        component = AzureOpenAIChatGenerator(
            azure_endpoint=Secret.from_env_var("AZURE_OPENAI_ENDPOINT"),
            api_version=Secret.from_env_var("AZURE_OPENAI_API_VERSION"),
        )
        init_params = component.to_dict()["init_parameters"]
        assert init_params["azure_endpoint"] == {
            "type": "env_var",
            "env_vars": ["AZURE_OPENAI_ENDPOINT"],
            "strict": True,
        }
        assert init_params["api_version"] == {
            "type": "env_var",
            "env_vars": ["AZURE_OPENAI_API_VERSION"],
            "strict": True,
        }

    def test_secret_azure_endpoint_and_api_version_roundtrip(self, monkeypatch):
        """Serializing and deserializing a component with Secret endpoint/version restores the Secrets."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.azure.openai.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        component = AzureOpenAIChatGenerator(
            azure_endpoint=Secret.from_env_var("AZURE_OPENAI_ENDPOINT"),
            api_version=Secret.from_env_var("AZURE_OPENAI_API_VERSION"),
        )
        deserialized = AzureOpenAIChatGenerator.from_dict(component.to_dict())
        assert deserialized.azure_endpoint == Secret.from_env_var("AZURE_OPENAI_ENDPOINT")
        assert deserialized.api_version == Secret.from_env_var("AZURE_OPENAI_API_VERSION")
        deserialized.warm_up()
        assert str(deserialized.client._azure_endpoint) == "https://test-resource.azure.openai.com/"
        assert deserialized.client._api_version == "2024-08-01-preview"

    def test_from_dict_with_secret_azure_endpoint_and_api_version(self, monkeypatch):
        """from_dict deserializes Secret azure_endpoint/api_version dicts and resolves them for the client."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.azure.openai.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        data = {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "azure_endpoint": {"env_vars": ["AZURE_OPENAI_ENDPOINT"], "strict": True, "type": "env_var"},
                "api_version": {"env_vars": ["AZURE_OPENAI_API_VERSION"], "strict": True, "type": "env_var"},
                "azure_deployment": "gpt-4.1-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "default_headers": {},
                "tools": None,
                "tools_strict": False,
                "azure_ad_token_provider": None,
                "http_client_kwargs": None,
            },
        }
        generator = AzureOpenAIChatGenerator.from_dict(data)
        # The Secret dicts are deserialized back into Secret objects
        assert generator.azure_endpoint == Secret.from_env_var("AZURE_OPENAI_ENDPOINT")
        assert generator.api_version == Secret.from_env_var("AZURE_OPENAI_API_VERSION")
        # And they are resolved to the string values the client expects
        generator.warm_up()
        assert str(generator.client._azure_endpoint) == "https://test-resource.azure.openai.com/"
        assert generator.client._api_version == "2024-08-01-preview"

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False, "type": "env_var"},
                "api_version": "2024-12-01-preview",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-4.1-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "default_headers": {},
                "tools": None,
                "tools_strict": False,
                "azure_ad_token_provider": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch, calendar_event_model):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            azure_ad_token=Secret.from_env_var("ENV_VAR1", strict=False),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            timeout=2.5,
            max_retries=10,
            generation_kwargs={
                "max_completion_tokens": 10,
                "some_test_param": "test-params",
                "response_format": calendar_event_model,
            },
            azure_ad_token_provider=default_azure_ad_token_provider,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure.AzureOpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "azure_ad_token": {"env_vars": ["ENV_VAR1"], "strict": False, "type": "env_var"},
                "api_version": "2024-12-01-preview",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-4.1-mini",
                "organization": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "timeout": 2.5,
                "max_retries": 10,
                "generation_kwargs": {
                    "max_completion_tokens": 10,
                    "some_test_param": "test-params",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "CalendarEvent",
                            "strict": True,
                            "schema": {
                                "properties": {
                                    "event_name": {"title": "Event Name", "type": "string"},
                                    "event_date": {"title": "Event Date", "type": "string"},
                                    "event_location": {"title": "Event Location", "type": "string"},
                                },
                                "required": ["event_name", "event_date", "event_location"],
                                "title": "CalendarEvent",
                                "type": "object",
                                "additionalProperties": False,
                            },
                        },
                    },
                },
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
                "api_version": "2024-12-01-preview",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-4.1-mini",
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
        assert generator.api_version == "2024-12-01-preview"
        assert generator.azure_endpoint == "some-non-existing-endpoint"
        assert generator.azure_deployment == "gpt-4.1-mini"
        assert generator.organization is None
        assert generator.streaming_callback is None
        assert generator.generation_kwargs == {}
        assert generator.timeout == 30.0
        assert generator.max_retries == 5
        assert generator.default_headers == {}
        assert generator.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert generator.tools_strict is False
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
                        "azure_deployment": "gpt-4.1-mini",
                        "organization": None,
                        "api_version": "2024-12-01-preview",
                        "streaming_callback": None,
                        "generation_kwargs": {},
                        "timeout": None,
                        "max_retries": None,
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
        assert "gpt-4.1-mini" in message.meta["model"]
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

    @pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY", None),
        reason="Export an env var called AZURE_OPENAI_API_KEY containing the Azure OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format(self):
        class CalendarEvent(BaseModel):
            event_name: str
            event_date: str
            event_location: str

        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = AzureOpenAIChatGenerator(
            api_version="2024-08-01-preview", generation_kwargs={"response_format": CalendarEvent}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

        assert message.meta["finish_reason"] == "stop"

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
                            "async_function": None,
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
    async def test_warm_up_async_builds_async_client(self, tools):
        component = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_completion_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
        )
        assert component.async_client is None
        await component.warm_up_async()
        assert component.async_client.api_key == "test-api-key"
        assert component.client is None
        assert component.azure_deployment == "gpt-4.1-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_completion_tokens": 10, "some_test_param": "test-params"}
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
        component = AzureOpenAIChatGenerator(generation_kwargs={"n": 1})
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4.1-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        await component.close_async()

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
        component = AzureOpenAIChatGenerator(tools=tools)
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
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

        await component.close_async()

    # additional tests intentionally omitted as they are covered by test_openai.py


@pytest.fixture
def mock_azure_clients(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    sync_cls = MagicMock(name="AzureOpenAI")
    async_cls = MagicMock(name="AsyncAzureOpenAI")
    async_cls.return_value.close = AsyncMock()
    monkeypatch.setattr(azure_chat_module, "AzureOpenAI", sync_cls)
    monkeypatch.setattr(azure_chat_module, "AsyncAzureOpenAI", async_cls)
    return sync_cls, async_cls


class TestComponentLifecycle:
    def test_warm_up_uses_default_timeout_and_max_retries(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        generator.warm_up()
        assert generator.client.max_retries == 5
        assert generator.client.timeout == 30.0

    def test_warm_up_uses_timeout_and_max_retries_from_parameters(self):
        generator = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("fake-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            timeout=40.0,
            max_retries=1,
        )
        generator.warm_up()
        assert generator.client.max_retries == 1
        assert generator.client.timeout == 40.0

    def test_warm_up_uses_timeout_and_max_retries_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        generator = AzureOpenAIChatGenerator(
            api_key=Secret.from_token("fake-api-key"), azure_endpoint="some-non-existing-endpoint"
        )
        generator.warm_up()
        assert generator.client.max_retries == 10
        assert generator.client.timeout == 100.0

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        with pytest.raises(OpenAIError):
            generator.warm_up()

    def test_warm_up_warms_tools_once(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        warm_up_calls = []

        class MockTool(Tool):
            def __init__(self, tool_name):
                super().__init__(
                    name=tool_name,
                    description=f"Mock tool {tool_name}",
                    parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                    function=lambda x: x,
                )

            def warm_up(self):
                warm_up_calls.append(self.name)

        generator = AzureOpenAIChatGenerator(
            azure_endpoint="some-non-existing-endpoint", tools=[MockTool("tool1"), MockTool("tool2")]
        )
        assert not generator._tools_warmed_up

        generator.warm_up()
        assert sorted(warm_up_calls) == ["tool1", "tool2"]
        assert generator._tools_warmed_up

        generator.warm_up()
        assert sorted(warm_up_calls) == ["tool1", "tool2"]

    def test_warm_up_with_no_tools_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        generator.warm_up()
        assert generator._tools_warmed_up

    def test_sync_lifecycle(self, mock_azure_clients):
        sync_cls, _ = mock_azure_clients
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert generator.client is None
        assert generator.async_client is None

        generator.warm_up()
        assert generator.client is sync_cls.return_value
        assert generator.async_client is None

        generator.close()
        sync_cls.return_value.close.assert_called_once()
        assert generator.client is None

    async def test_async_lifecycle(self, mock_azure_clients):
        _, async_cls = mock_azure_clients
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")

        await generator.warm_up_async()
        assert generator.async_client is async_cls.return_value
        assert generator.client is None

        await generator.close_async()
        async_cls.return_value.close.assert_awaited_once()
        assert generator.async_client is None

    async def test_close_is_safe_without_warm_up(self, mock_azure_clients):
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        generator.close()
        await generator.close_async()
        assert generator.client is None
        assert generator.async_client is None

    async def test_close_and_close_async_are_independent(self, mock_azure_clients):
        generator = AzureOpenAIChatGenerator(azure_endpoint="some-non-existing-endpoint")
        generator.warm_up()
        await generator.warm_up_async()

        generator.close()
        assert generator.client is None
        assert generator.async_client is not None

        await generator.close_async()
        assert generator.async_client is None
