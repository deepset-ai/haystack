# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat.openai_responses import OpenAIResponsesChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, ImageContent, StreamingChunk, TextContent, ToolCall
from haystack.tools import ComponentTool, Tool, Toolset, create_tool_from_function
from haystack.utils import Secret


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


@pytest.fixture
def calendar_event_model():
    return CalendarEvent


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


def weather_function(city: str) -> dict[str, Any]:
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


@pytest.fixture
def tools():
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather_function,
    )

    # We add a tool that has a more complex parameter signature
    message_extractor_tool = ComponentTool(
        component=MessageExtractor(),
        name="message_extractor",
        description="Useful for returning the text content of ChatMessage objects",
    )
    return [weather_tool, message_extractor_tool]


# Tool Function used in the test_live_run_with_agent_streaming_and_reasoning test
def calculate(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


class RecordingCallback:
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.tool_calls = []
        self.counter = 0

    def __call__(self, chunk: StreamingChunk):
        self.counter += 1
        if chunk.content:
            self.content += chunk.content
        if chunk.reasoning:
            self.reasoning += chunk.reasoning.reasoning_text
        if chunk.tool_calls:
            self.tool_calls.extend(chunk.tool_calls)


class TestInitialization:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIResponsesChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-5-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.timeout == 30
        assert component.client.max_retries == 5
        assert component.tools is None
        assert not component.tools_strict
        assert component.http_client_kwargs is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIResponsesChatGenerator()

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=40.0,
            max_retries=1,
            tools=[tool],
            tools_strict=True,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 40.0
        assert component.client.max_retries == 1
        assert component.tools == [tool]
        assert component.tools_strict
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10

    def test_init_with_toolset(self, tools, monkeypatch):
        """Test that the OpenAIChatGenerator can be initialized with a Toolset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = OpenAIResponsesChatGenerator(tools=toolset)
        assert generator.tools == toolset


class TestSerDe:
    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIResponsesChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-5-mini",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {},
                "tools": None,
                "tools_strict": False,
                "max_retries": None,
                "timeout": None,
                "http_client_kwargs": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch, calendar_event_model):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="gpt-5-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params", "text_format": calendar_event_model},
            tools=[tool],
            tools_strict=True,
            max_retries=10,
            timeout=100.0,
            http_client_kwargs={"proxy": "http://example.com:8080", "verify": False},
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "gpt-5-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "max_retries": 10,
                "timeout": 100.0,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
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
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "inputs_from_state": None,
                            "name": "name",
                            "outputs_to_state": None,
                            "outputs_to_string": None,
                            "parameters": {"x": {"type": "string"}},
                        },
                    }
                ],
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-5-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "max_retries": 10,
                "timeout": 100.0,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
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
                "tools_strict": True,
                "http_client_kwargs": {"proxy": "http://example.com:8080", "verify": False},
            },
        }
        component = OpenAIResponsesChatGenerator.from_dict(data)

        assert isinstance(component, OpenAIResponsesChatGenerator)
        assert component.model == "gpt-5-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]
        assert component.tools_strict
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10
        assert component.http_client_kwargs == {"proxy": "http://example.com:8080", "verify": False}

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-5-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": None,
            },
        }
        with pytest.raises(ValueError):
            OpenAIResponsesChatGenerator.from_dict(data)

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        """Test that the OpenAIChatGenerator can be deserialized from a dictionary with a Toolset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = OpenAIResponsesChatGenerator(tools=toolset)
        data = component.to_dict()

        deserialized_component = OpenAIResponsesChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, Toolset)
        assert len(deserialized_component.tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component.tools)


class TestWarmUp:
    def test_warm_up_with_tools(self, monkeypatch):
        """Test that warm_up() calls warm_up on tools and is idempotent."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        # Create a mock tool that tracks if warm_up() was called
        class MockTool(Tool):
            warm_up_call_count = 0  # Class variable to track calls

            def __init__(self):
                super().__init__(
                    name="mock_tool",
                    description="A mock tool for testing",
                    parameters={"x": {"type": "string"}},
                    function=lambda x: x,
                )

            def warm_up(self):
                MockTool.warm_up_call_count += 1

        # Reset the class variable before test
        MockTool.warm_up_call_count = 0
        mock_tool = MockTool()

        # Create OpenAIChatGenerator with the mock tool
        component = OpenAIResponsesChatGenerator(tools=[mock_tool])

        # Verify initial state - warm_up not called yet
        assert MockTool.warm_up_call_count == 0
        assert not component._is_warmed_up

        # Call warm_up() on the generator
        component.warm_up()

        # Assert that the tool's warm_up() was called
        assert MockTool.warm_up_call_count == 1
        assert component._is_warmed_up

        component.warm_up()

        # The tool's warm_up should still only have been called once
        assert MockTool.warm_up_call_count == 1
        assert component._is_warmed_up

    def test_warm_up_with_no_tools(self, monkeypatch):
        """Test that warm_up() works when no tools are provided."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        component = OpenAIResponsesChatGenerator()

        # Verify initial state
        assert not component._is_warmed_up
        assert component.tools is None

        # Verify the component is warmed up
        component.warm_up()
        assert component._is_warmed_up

    def test_warm_up_with_openai_tools(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        component = OpenAIResponsesChatGenerator(
            tools=[
                {"type": "web_search_preview"},
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_description": "A Dungeons and Dragons MCP server to assist with dice rolling.",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                    "require_approval": "never",
                },
            ]
        )

        # Make sure the component can still be warmed up even when using openai tools
        assert not component._is_warmed_up
        component.warm_up()
        assert component._is_warmed_up

    def test_warm_up_with_multiple_tools(self, monkeypatch):
        """Test that warm_up() works with multiple tools."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        from haystack.tools import Tool

        # Track warm_up calls
        warm_up_calls = []

        class MockTool(Tool):
            def __init__(self, tool_name):
                super().__init__(
                    name=tool_name,
                    description=f"Mock tool {tool_name}",
                    parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                    function=lambda x: f"{tool_name} result: {x}",
                )

            def warm_up(self):
                warm_up_calls.append(self.name)

        mock_tool1 = MockTool("tool1")
        mock_tool2 = MockTool("tool2")

        # Use a LIST of tools, not a Toolset
        component = OpenAIResponsesChatGenerator(tools=[mock_tool1, mock_tool2])

        # Assert that both tools' warm_up() were called
        component.warm_up()
        assert "tool1" in warm_up_calls
        assert "tool2" in warm_up_calls
        assert component._is_warmed_up

        # Verify idempotency
        call_count = len(warm_up_calls)
        component.warm_up()
        assert len(warm_up_calls) == call_count


class TestRun:
    def test_run_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            chat_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
            component = OpenAIResponsesChatGenerator(tools=duplicate_tools)
            component.run(chat_messages)

    def test_run_with_wrong_model(self):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("Invalid model name")

        generator = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="something-obviously-wrong"
        )

        generator.client = mock_client

        with pytest.raises(OpenAIError):
            generator.run([ChatMessage.from_user("irrelevant")])

    def test_run(self, openai_mock_responses, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-4", generation_kwargs={"include": ["message.output_text.logprobs"]}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-5" in message.meta["model"]
        assert message.meta["usage"]["total_tokens"] > 0
        assert message.meta["id"] is not None

    def test_run_with_flattened_generation_kwargs(self, openai_mock_responses, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-4",
            generation_kwargs={"reasoning_effort": "low", "reasoning_summary": "auto", "verbosity": "low"},
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        assert openai_mock_responses.call_args.kwargs["reasoning"] == {"effort": "low", "summary": "auto"}
        assert openai_mock_responses.call_args.kwargs["text"] == {"verbosity": "low"}

    def test_run_with_params_streaming(self, openai_mock_responses_stream_text_delta):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        response = component.run([ChatMessage.from_user("What's the capital of France")])

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "The capital of France is Paris." in response["replies"][0].text

    def test_run_with_params_streaming_reasoning_summary_delta(self, openai_mock_responses_reasoning_summary_delta):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )
        response = component.run(
            [ChatMessage.from_user("What's the capital of France")],
            generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}},
        )

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        print(response["replies"])
        assert len(response["replies"]) == 1
        assert "I need to check the capital of France." in response["replies"][0].reasoning.reasoning_text


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
@pytest.mark.integration
class TestIntegration:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-4.1-nano", generation_kwargs={"include": ["message.output_text.logprobs"]}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4.1-nano" in message.meta["model"]
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["total_tokens"] > 0
        assert message.meta["id"] is not None
        assert message.meta["logprobs"] is not None

    def test_live_run_with_reasoning(self):
        chat_messages = [ChatMessage.from_user("Explain in 2 lines why is there a Moon?")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-5-nano", generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.reasoning is not None
        assert "moon" in message.text.lower() or "moon" in message.reasoning.reasoning_text.lower()
        assert "gpt-5-nano" in message.meta["model"]
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0
        assert "reasoning_tokens" in message.meta["usage"]["output_tokens_details"]

    def test_live_run_with_text_format(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = OpenAIResponsesChatGenerator(
            model="gpt-5-nano", generation_kwargs={"text_format": calendar_event_model}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        print(message.text)
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

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
        component = OpenAIResponsesChatGenerator(model="gpt-5-nano", generation_kwargs={"text": json_schema})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Jane" in msg["name"]
        assert msg["age"] == 54
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0

    @pytest.mark.skip(
        reason="Streaming plus pydantic based model does not work due to known issue in openai python "
        "sdk https://github.com/openai/openai-python/issues/2305"
    )
    def test_live_run_with_text_format_and_streaming(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = OpenAIResponsesChatGenerator(
            streaming_callback=print_streaming_chunk, generation_kwargs={"text_format": calendar_event_model}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

    def test_live_run_streaming(self):
        callback = RecordingCallback()
        component = OpenAIResponsesChatGenerator(
            model="gpt-4.1-nano",
            streaming_callback=callback,
            generation_kwargs={"include": ["message.output_text.logprobs"]},
        )
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        # Basic response checks
        assert "replies" in results
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert isinstance(message.meta, dict)

        # Metadata checks
        metadata = message.meta
        assert "gpt-4.1-nano" in metadata["model"]
        assert metadata["logprobs"] is not None
        # Usage information checks
        assert isinstance(metadata.get("usage"), dict), "meta.usage not a dict"
        usage = metadata["usage"]
        assert "output_tokens" in usage and usage["output_tokens"] > 0

        # Detailed token information checks
        assert isinstance(usage.get("output_tokens_details"), dict), "usage.output_tokens_details not a dict"

        # Streaming callback verification
        assert callback.counter > 1
        assert "Paris" in callback.content

    def test_live_run_with_reasoning_and_streaming(self):
        callback = RecordingCallback()
        chat_messages = [ChatMessage.from_user("Explain in 2 lines why is there a Moon?")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-5-nano",
            generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}},
            streaming_callback=callback,
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert callback.reasoning == message.reasoning.reasoning_text
        assert "moon" in callback.content.lower() or "moon" in callback.reasoning.lower()
        assert "gpt-5-nano" in message.meta["model"]
        assert message.reasonings is not None
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0
        assert "reasoning_tokens" in message.meta["usage"]["output_tokens_details"]

    def test_live_run_with_tools_streaming(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]

        component = OpenAIResponsesChatGenerator(
            model="gpt-5-nano", tools=tools, streaming_callback=print_streaming_chunk
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_calls = message.tool_calls
        assert len(tool_calls) == 2

        for tool_call in tool_calls:
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        # Extract city names (handle cases like "Berlin, Germany" -> "Berlin")
        city_values = [arg["city"].split(",")[0].strip().lower() for arg in arguments]
        assert "berlin" in city_values and "paris" in city_values
        assert len(city_values) == 2

    def test_live_run_with_toolset(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        toolset = Toolset(tools)
        component = OpenAIResponsesChatGenerator(model="gpt-5-nano", tools=toolset)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments.keys() == {"city"}
        assert "Paris" in tool_call.arguments["city"]

    def test_live_run_multimodal(self, test_files_path):
        image_path = test_files_path / "images" / "apple.jpg"

        # we resize the image to keep this test fast (around 1s) - increase the size in case of errors
        image_content = ImageContent.from_file_path(file_path=image_path, size=(100, 100), detail="low")

        chat_messages = [ChatMessage.from_user(content_parts=["What does this image show? Max 5 words", image_content])]

        generator = OpenAIResponsesChatGenerator(model="gpt-5-nano")
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.text
        assert "apple" in message.text.lower()

        assert message.is_from(ChatRole.ASSISTANT)
        assert not message.tool_calls
        assert not message.tool_call_results

    @pytest.mark.skip(reason="The tool calls time out resulting in failing")
    def test_live_run_with_openai_tools(self):
        """
        Test the use of generator with a list of OpenAI tools and MCP tools.
        """
        chat_messages = [ChatMessage.from_user("What was a positive news story from today?")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-5",
            tools=[
                {"type": "web_search_preview"},
                {
                    "type": "mcp",
                    "server_label": "dmcp",
                    "server_description": "A Dungeons and Dragons MCP server to assist with dice rolling.",
                    "server_url": "https://dmcp-server.deno.dev/sse",
                    "require_approval": "never",
                },
            ],
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.meta["status"] == "completed"

        chat_messages = [ChatMessage.from_user("Roll 2d4+1")]
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.meta["status"] == "completed"

    def test_live_run_with_tools_streaming_and_reasoning(self, tools):
        chat_messages = [
            ChatMessage.from_user("What's the weather like in Paris and Berlin? Make sure to use the provided tool.")
        ]

        component = OpenAIResponsesChatGenerator(
            model="gpt-5-nano",
            tools=tools,
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}},
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.reasonings is not None
        # model sometimes skips reasoning
        # needs to be cross checked
        assert message.reasonings[0].extra is not None
        assert not message.text
        assert message.tool_calls
        tool_calls = message.tool_calls
        assert len(tool_calls) > 0

        for tool_call in tool_calls:
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    def test_live_run_with_agent_streaming_and_reasoning(self):
        # Tool Definition
        calculator_tool = Tool(
            name="calculator",
            description="Evaluate basic math expressions.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression to evaluate"}},
                "required": ["expression"],
            },
            function=calculate,
            outputs_to_state={"calc_result": {"source": "result"}},
        )

        # Agent Setup
        agent = Agent(
            chat_generator=OpenAIResponsesChatGenerator(
                model="gpt-5-nano",
                tools_strict=True,
                generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}},
            ),
            streaming_callback=print_streaming_chunk,
            tools=[calculator_tool],
            exit_conditions=["text"],
            state_schema={"calc_result": {"type": int}},
        )

        # Run the Agent
        agent.warm_up()
        response = agent.run(
            messages=[
                ChatMessage.from_user("What is 7 * (4 + 2)? Make sure to call the calculator tool to get the answer.")
            ]
        )

        tool_call_results = []
        tool_calls = []

        for message in response["messages"]:
            if message.tool_call_results is not None:
                tool_call_results.extend(message.tool_call_results)
            if message.tool_calls is not None:
                tool_calls.extend(message.tool_calls)

        assert len(tool_calls) > 0
        assert len(tool_call_results) > 0

        # Verify state was updated
        assert "calc_result" in response
        assert response["messages"][-1].text is not None

    def test_live_run_agent_with_images_in_tool_result(self, test_files_path):
        def retrieve_image():
            return [
                TextContent("Here is the retrieved image."),
                ImageContent.from_file_path(test_files_path / "images" / "apple.jpg", size=(100, 100), detail="low"),
            ]

        image_retriever_tool = create_tool_from_function(
            name="retrieve_image",
            description="Tool to retrieve an image",
            function=retrieve_image,
            outputs_to_string={"raw_result": True},
        )

        agent = Agent(
            chat_generator=OpenAIResponsesChatGenerator(model="gpt-5-nano"),
            system_prompt="You are an Agent that can retrieve images and describe them.",
            tools=[image_retriever_tool],
        )

        user_message = ChatMessage.from_user("Retrieve the image and describe it in max 5 words.")
        result = agent.run(messages=[user_message])

        assert "apple" in result["last_message"].text.lower()


class TestOpenAIResponsesChatGeneratorAsync:
    def test_init_should_also_create_async_client_with_same_args(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            api_base_url="test-base-url",
            organization="test-organization",
            timeout=30,
            max_retries=5,
        )

        assert isinstance(component.async_client, AsyncOpenAI)
        assert component.async_client.api_key == "test-api-key"
        assert component.async_client.organization == "test-organization"
        assert component.async_client.base_url == "test-base-url/"
        assert component.async_client.timeout == 30
        assert component.async_client.max_retries == 5

    @pytest.mark.asyncio
    async def test_run_async(self, openai_mock_async_responses):
        component = OpenAIResponsesChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = await component.run_async([ChatMessage.from_user("What's the capital of France")])

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator(
            model="gpt-4.1-nano", generation_kwargs={"include": ["message.output_text.logprobs"]}
        )
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4.1-nano" in message.meta["model"]
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["total_tokens"] > 0
        assert message.meta["id"] is not None
        assert message.meta["logprobs"] is not None
