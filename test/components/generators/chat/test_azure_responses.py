# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any

import pytest
from pydantic import BaseModel

from haystack import Pipeline, component
from haystack.components.generators.chat import AzureOpenAIResponsesChatGenerator
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


class TestInitialization:
    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS is a non-empty list of strings."""
        models = AzureOpenAIResponsesChatGenerator.SUPPORTED_MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client is None
        assert component.async_client is None
        assert component._azure_deployment == "gpt-5-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_azure_endpoint(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(ValueError):
            AzureOpenAIResponsesChatGenerator()

    def test_init_with_parameters(self, tools):
        component = AzureOpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_completion_tokens": 10, "some_test_param": "test-params"},
            tools=tools,
            tools_strict=True,
        )
        assert component.client is None
        assert component.async_client is None
        assert component._azure_deployment == "gpt-5-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_completion_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == tools
        assert component.tools_strict
        assert component.max_retries is None

    def test_init_with_toolset(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be initialized with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
        assert generator.tools == toolset


class TestSerDe:
    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-5-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch, calendar_event_model):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            azure_endpoint="some-non-existing-endpoint",
            streaming_callback=print_streaming_chunk,
            timeout=2.5,
            max_retries=10,
            generation_kwargs={
                "max_completion_tokens": 10,
                "some_test_param": "test-params",
                "text_format": calendar_event_model,
            },
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-5-mini",
                "organization": None,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "timeout": 2.5,
                "max_retries": 10,
                "generation_kwargs": {
                    "max_completion_tokens": 10,
                    "some_test_param": "test-params",
                    "text": {
                        "format": {
                            "type": "json_schema",
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
                        }
                    },
                },
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }

    def test_to_dict_with_ad_token_provider(self):
        component = AzureOpenAIResponsesChatGenerator(
            api_key=default_azure_ad_token_provider, azure_endpoint="some-non-existing-endpoint"
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": "haystack.utils.azure.default_azure_ad_token_provider",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-5-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be serialized to a dictionary with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools[:1])
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
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
                            "function": "generators.chat.test_azure_responses.get_weather",
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

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "test-ad-token")
        data = {
            "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False, "type": "env_var"},
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-5-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": 30.0,
                "max_retries": 5,
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

        generator = AzureOpenAIResponsesChatGenerator.from_dict(data)
        assert isinstance(generator, AzureOpenAIResponsesChatGenerator)

        assert generator.api_key == Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False)
        assert generator._azure_endpoint == "some-non-existing-endpoint"
        assert generator._azure_deployment == "gpt-5-mini"
        assert generator.organization is None
        assert generator.streaming_callback is None
        assert generator.generation_kwargs == {}
        assert generator.timeout == 30.0
        assert generator.max_retries == 5
        assert generator.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert generator.tools_strict is False
        assert generator.http_client_kwargs is None

    def test_from_dict_with_ad_token_provider(self):
        data = {
            "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": "haystack.utils.azure.default_azure_ad_token_provider",
                "azure_endpoint": "some-non-existing-endpoint",
                "azure_deployment": "gpt-5-mini",
                "organization": None,
                "streaming_callback": None,
                "generation_kwargs": {},
                "timeout": None,
                "max_retries": None,
                "tools": None,
                "tools_strict": False,
                "http_client_kwargs": None,
            },
        }

        generator = AzureOpenAIResponsesChatGenerator.from_dict(data)
        assert isinstance(generator, AzureOpenAIResponsesChatGenerator)

        assert generator.api_key == default_azure_ad_token_provider
        assert generator._azure_endpoint == "some-non-existing-endpoint"
        assert generator._azure_deployment == "gpt-5-mini"
        assert generator.organization is None
        assert generator.streaming_callback is None
        assert generator.generation_kwargs == {}
        assert generator.timeout is None
        assert generator.max_retries is None
        assert generator.tools is None
        assert generator.tools_strict is False
        assert generator.http_client_kwargs is None

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        """Test that the AzureOpenAIChatGenerator can be deserialized from a dictionary with a Toolset."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint", tools=toolset)
        data = component.to_dict()

        deserialized_component = AzureOpenAIResponsesChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)

    def test_pipeline_serialization_deserialization(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        generator = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        p = Pipeline()
        p.add_component(instance=generator, name="generator")

        assert p.to_dict() == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack.components.generators.chat.azure_responses.AzureOpenAIResponsesChatGenerator",
                    "init_parameters": {
                        "azure_endpoint": "some-non-existing-endpoint",
                        "azure_deployment": "gpt-5-mini",
                        "organization": None,
                        "streaming_callback": None,
                        "generation_kwargs": {},
                        "timeout": None,
                        "max_retries": None,
                        "api_key": {"type": "env_var", "env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False},
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                }
            },
            "connections": [],
        }
        p_str = p.dumps()
        q = Pipeline.loads(p_str)
        assert p.to_dict() == q.to_dict()


class TestComponentLifecycle:
    def test_warm_up_warms_tools_once(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
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

        component = AzureOpenAIResponsesChatGenerator(
            azure_endpoint="some-non-existing-endpoint", tools=[MockTool("tool1"), MockTool("tool2")]
        )
        assert not component._tools_warmed_up

        component.warm_up()
        assert sorted(warm_up_calls) == ["tool1", "tool2"]
        assert component._tools_warmed_up

        component.warm_up()
        assert sorted(warm_up_calls) == ["tool1", "tool2"]

    def test_warm_up_with_no_tools_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        component.warm_up()
        assert component._tools_warmed_up

    def test_sync_lifecycle(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        assert component.client is None
        assert component.async_client is None

        component.warm_up()
        assert component.client is not None
        assert component.async_client is None

        component.close()
        assert component.client is None

    async def test_async_lifecycle(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")

        await component.warm_up_async()
        assert component.async_client is not None
        assert component.client is None

        await component.close_async()
        assert component.async_client is None

    async def test_close_is_safe_without_warm_up(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
        component = AzureOpenAIResponsesChatGenerator(azure_endpoint="some-non-existing-endpoint")
        component.close()
        await component.close_async()
        assert component.client is None
        assert component.async_client is None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AZURE_OPENAI_API_KEY", None) or not os.environ.get("AZURE_OPENAI_ENDPOINT", None),
    reason=(
        "Please export env variables called AZURE_OPENAI_API_KEY containing "
        "the Azure OpenAI key, AZURE_OPENAI_ENDPOINT containing "
        "the Azure OpenAI endpoint URL to run this test."
    ),
)
class TestIntegration:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AzureOpenAIResponsesChatGenerator(azure_deployment="gpt-4o-mini")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "paris" in message.text.lower()
        assert "gpt-4o-mini" in message.meta["model"]
        assert message.meta["status"] == "completed"

    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AzureOpenAIResponsesChatGenerator(
            organization="HaystackCI", tools=tools, azure_deployment="gpt-4o-mini"
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert "city" in tool_call.arguments
        assert "paris" in tool_call.arguments["city"].lower()
        assert message.meta["status"] == "completed"

    def test_live_run_with_text_format(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = AzureOpenAIResponsesChatGenerator(
            azure_deployment="gpt-4o-mini", generation_kwargs={"text_format": calendar_event_model}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "marketing summit" in msg["event_name"].lower()
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)
        assert message.meta["status"] == "completed"

    # So far from documentation, responses.parse only supports BaseModel
    def test_live_run_with_text_format_json_schema(self):
        json_schema = {
            "format": {
                "type": "json_schema",
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "number", "minimum": 0, "maximum": 130},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            }
        }
        chat_messages = [ChatMessage.from_user("Jane 54 years old")]
        component = AzureOpenAIResponsesChatGenerator(
            azure_deployment="gpt-4o-mini", generation_kwargs={"text": json_schema}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "jane" in msg["name"].lower()
        assert msg["age"] == 54
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0


class TestAzureOpenAIResponsesChatGeneratorAsync:
    async def test_warm_up_async_creates_async_client_with_expected_args(self, tools):
        component = AzureOpenAIResponsesChatGenerator(
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
        assert component._azure_deployment == "gpt-5-mini"
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
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = AzureOpenAIResponsesChatGenerator(azure_deployment="gpt-4o-mini")
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "paris" in message.text.lower()
        assert "gpt-4o-mini" in message.meta["model"]
        assert message.meta["status"] == "completed"

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
        component = AzureOpenAIResponsesChatGenerator(tools=tools, azure_deployment="gpt-4o-mini")
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert "city" in tool_call.arguments
        assert "paris" in tool_call.arguments["city"].lower()
        assert message.meta["status"] == "completed"

    # additional tests intentionally omitted as they are covered by test_openai_responses.py
    # and test_openai_responses_conversion.py
