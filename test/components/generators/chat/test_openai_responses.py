# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any, Optional
from unittest.mock import ANY, MagicMock

import pytest
from openai import OpenAIError
from openai.types import Reasoning, ResponseFormatText
from openai.types.responses import (
    FunctionTool,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat.openai_responses import (
    OpenAIResponsesChatGenerator,
    _convert_chat_message_to_responses_api_format,
    _convert_response_chunk_to_streaming_chunk,
)
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import ComponentTool, Tool, Toolset
from haystack.utils import Secret


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


@pytest.fixture
def calendar_event_model():
    return CalendarEvent


def callback(chunk: StreamingChunk) -> None: ...


@component
class MessageExtractor:
    @component.output_types(messages=list[str], meta=dict[str, Any])
    def run(self, messages: list[ChatMessage], meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


class TestOpenAIResponsesChatGenerator:
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

    def test_run_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            chat_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
            component = OpenAIResponsesChatGenerator(tools=duplicate_tools)
            component.run(chat_messages)

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

    def test_chat_generator_with_toolset_initialization(self, tools, monkeypatch):
        """Test that the OpenAIChatGenerator can be initialized with a Toolset."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = OpenAIResponsesChatGenerator(tools=toolset)
        assert generator.tools == toolset

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

    def test_convert_chat_message_to_responses_api_format(self):
        chat_message = ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=[
                ReasoningContent(
                    reasoning_text="I need to use the functions.weather tool.",
                    extra={"id": "rs_0d13efdd", "type": "reasoning"},
                ),
                ToolCall(
                    tool_name="weather",
                    arguments={"location": "Berlin"},
                    id="fc_0d13efdd",
                    extra={"call_id": "call_a82vwFAIzku9SmBuQuecQSRq"},
                ),
            ],
            _name=None,
            # some keys are removed to keep the test concise
            _meta={
                "id": "resp_0d13efdd97aa4",
                "created_at": 1761148307.0,
                "model": "gpt-5-mini-2025-08-07",
                "object": "response",
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [
                    {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                        "type": "function",
                        "description": "A tool to get the weather",
                    }
                ],
                "top_p": 1.0,
                "reasoning": {"effort": "low", "summary": "detailed"},
                "usage": {"input_tokens": 59, "output_tokens": 19, "total_tokens": 78},
                "store": True,
            },
        )
        responses_api_format = _convert_chat_message_to_responses_api_format(chat_message)
        assert responses_api_format == [
            {
                "id": "rs_0d13efdd",
                "type": "reasoning",
                "summary": [{"text": "I need to use the functions.weather tool.", "type": "summary_text"}],
            },
            {
                "type": "function_call",
                "name": "weather",
                "arguments": '{"location": "Berlin"}',
                "id": "fc_0d13efdd",
                "call_id": "call_a82vwFAIzku9SmBuQuecQSRq",
            },
        ]

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

    def test_warm_up_with_no_openai_tools(self, monkeypatch):
        """Test that warm_up() works when no tools are provided."""
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["total_tokens"] > 0
        assert message.meta["id"] is not None

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_reasoning(self):
        chat_messages = [ChatMessage.from_user("Explain in 2 lines why is there a Moon?")]
        component = OpenAIResponsesChatGenerator(generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Moon" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.reasonings is not None
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0
        assert "reasoning_tokens" in message.meta["usage"]["output_tokens_details"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_text_format(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = OpenAIResponsesChatGenerator(generation_kwargs={"text_format": calendar_event_model})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
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
        component = OpenAIResponsesChatGenerator(generation_kwargs={"text": json_schema})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Jane" in msg["name"]
        assert msg["age"] == 54
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_ser_deser_and_text_format(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = OpenAIResponsesChatGenerator(generation_kwargs={"text_format": calendar_event_model})
        serialized = component.to_dict()
        deser = OpenAIResponsesChatGenerator.from_dict(serialized)
        results = deser.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

    def test_run_with_wrong_model(self):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = OpenAIError("Invalid model name")

        generator = OpenAIResponsesChatGenerator(
            api_key=Secret.from_token("test-api-key"), model="something-obviously-wrong"
        )

        generator.client = mock_client

        with pytest.raises(OpenAIError):
            generator.run([ChatMessage.from_user("irrelevant")])

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = OpenAIResponsesChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        # Basic response checks
        assert "replies" in results
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert isinstance(message.meta, dict)

        # Metadata checks
        metadata = message.meta
        assert "gpt-5-mini" in metadata["model"]

        # Usage information checks
        assert isinstance(metadata.get("usage"), dict), "meta.usage not a dict"
        usage = metadata["usage"]
        assert "output_tokens" in usage and usage["output_tokens"] > 0

        # Detailed token information checks
        assert isinstance(usage.get("output_tokens_details"), dict), "usage.output_tokens_details not a dict"

        # Streaming callback verification
        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_reasoning_and_streaming(self):
        class Callback:
            def __init__(self):
                self.reasoning_content = ""

            def __call__(self, chunk: StreamingChunk) -> None:
                self.reasoning_content += chunk.reasoning.reasoning_text if chunk.reasoning else ""

        chat_messages = [ChatMessage.from_user("Explain in 2 lines why is there a Moon?")]
        callback = Callback()
        component = OpenAIResponsesChatGenerator(
            generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}}, streaming_callback=callback
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert callback.reasoning_content == message.reasoning.reasoning_text
        assert "Moon" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.reasonings is not None
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["output_tokens"] > 0
        assert "reasoning_tokens" in message.meta["usage"]["output_tokens_details"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]

        component = OpenAIResponsesChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
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
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_toolset(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        toolset = Toolset(tools)
        component = OpenAIResponsesChatGenerator(tools=toolset)
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming_and_reasoning(self, tools):
        chat_messages = [
            ChatMessage.from_user("What's the weather like in Paris and Berlin? Make sure to use the provided tool.")
        ]

        component = OpenAIResponsesChatGenerator(
            model="gpt-5",
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_agent_streaming_and_reasoning(self):
        # Tool Function
        def calculate(expression: str) -> dict:
            try:
                result = eval(expression, {"__builtins__": {}})
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

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
                tools_strict=True, generation_kwargs={"reasoning": {"summary": "auto", "effort": "low"}}
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


class TestConvertResponseChunkToStreamingChunk:
    def test_convert_only_text(self):
        openai_chunks = [
            ResponseCreatedEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    service_tier="auto",
                    status="in_progress",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=0,
                type="response.created",
            ),
            ResponseInProgressEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    service_tier="auto",
                    status="in_progress",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=1,
                type="response.in_progress",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseReasoningItem(
                    id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=2,
                type="response.output_item.added",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=3,
                type="response.output_item.done",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseOutputMessage(
                    id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    content=[],
                    role="assistant",
                    status="in_progress",
                    type="message",
                ),
                output_index=1,
                sequence_number=4,
                type="response.output_item.added",
            ),
            ResponseContentPartAddedEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                output_index=1,
                part=ResponseOutputText(annotations=[], text="", type="output_text", logprobs=[]),
                sequence_number=5,
                type="response.content_part.added",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="Germany",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=6,
                type="response.output_text.delta",
                obfuscation="EV5gCoyiD",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=":",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=7,
                type="response.output_text.delta",
                obfuscation="EkdNXp1EE2Cgj8z",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=" Berlin",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=8,
                type="response.output_text.delta",
                obfuscation="1eS0q9aye",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="\n",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=9,
                type="response.output_text.delta",
                obfuscation="H9Ict3F41DwGS4a",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="France",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=10,
                type="response.output_text.delta",
                obfuscation="4vxrblWURx",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=":",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=11,
                type="response.output_text.delta",
                obfuscation="B1CMJsNGhhqIz5K",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=" Paris",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=12,
                type="response.output_text.delta",
                obfuscation="ojbz89bS7j",
            ),
            ResponseTextDoneEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=13,
                text="Germany: Berlin\nFrance: Paris",
                type="response.output_text.done",
            ),
            ResponseContentPartDoneEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                output_index=1,
                part=ResponseOutputText(
                    annotations=[], text="Germany: Berlin\nFrance: Paris", type="output_text", logprobs=[]
                ),
                sequence_number=14,
                type="response.content_part.done",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseOutputMessage(
                    id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="Germany: Berlin\nFrance: Paris", type="output_text", logprobs=[]
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=1,
                sequence_number=15,
                type="response.output_item.done",
            ),
            ResponseCompletedEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    error=None,
                    incomplete_details=None,
                    instructions=None,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[
                        ResponseReasoningItem(
                            id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                        ),
                        ResponseOutputMessage(
                            id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                            content=[
                                ResponseOutputText(
                                    annotations=[],
                                    text="Germany: Berlin\nFrance: Paris",
                                    type="output_text",
                                    logprobs=[],
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                        ),
                    ],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    safety_identifier=None,
                    service_tier="default",
                    status="completed",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    usage=ResponseUsage(
                        input_tokens=15,
                        input_tokens_details=InputTokensDetails(cached_tokens=0),
                        output_tokens=77,
                        output_tokens_details=OutputTokensDetails(reasoning_tokens=64),
                        total_tokens=92,
                    ),
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=16,
                type="response.completed",
            ),
        ]
        streaming_chunks = []
        for chunk in openai_chunks:
            streaming_chunk = _convert_response_chunk_to_streaming_chunk(chunk, previous_chunks=streaming_chunks)
            streaming_chunks.append(streaming_chunk)

        assert streaming_chunks == [
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "service_tier": "auto",
                        "status": "in_progress",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 0,
                    "type": "response.created",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "service_tier": "auto",
                        "status": "in_progress",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 1,
                    "type": "response.in_progress",
                },
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=0,
                start=True,
                reasoning=ReasoningContent(
                    reasoning_text="",
                    extra={
                        "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                        "summary": [],
                        "type": "reasoning",
                    },
                ),
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                        "summary": [],
                        "type": "reasoning",
                    },
                    "output_index": 0,
                    "sequence_number": 3,
                    "type": "response.output_item.done",
                },
                index=0,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                        "content": [],
                        "role": "assistant",
                        "status": "in_progress",
                        "type": "message",
                    },
                    "output_index": 1,
                    "sequence_number": 4,
                    "type": "response.output_item.added",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "output_index": 1,
                    "part": {"annotations": [], "text": "", "type": "output_text", "logprobs": []},
                    "sequence_number": 5,
                    "type": "response.content_part.added",
                },
                index=1,
            ),
            StreamingChunk(
                content="Germany",
                meta={
                    "content_index": 0,
                    "delta": "Germany",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 6,
                    "type": "response.output_text.delta",
                    "obfuscation": "EV5gCoyiD",
                    "received_at": ANY,
                },
                index=1,
                start=True,
            ),
            StreamingChunk(
                content=":",
                meta={
                    "content_index": 0,
                    "delta": ":",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 7,
                    "type": "response.output_text.delta",
                    "obfuscation": "EkdNXp1EE2Cgj8z",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=" Berlin",
                meta={
                    "content_index": 0,
                    "delta": " Berlin",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 8,
                    "type": "response.output_text.delta",
                    "obfuscation": "1eS0q9aye",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="\n",
                meta={
                    "content_index": 0,
                    "delta": "\n",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 9,
                    "type": "response.output_text.delta",
                    "obfuscation": "H9Ict3F41DwGS4a",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="France",
                meta={
                    "content_index": 0,
                    "delta": "France",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 10,
                    "type": "response.output_text.delta",
                    "obfuscation": "4vxrblWURx",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=":",
                meta={
                    "content_index": 0,
                    "delta": ":",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 11,
                    "type": "response.output_text.delta",
                    "obfuscation": "B1CMJsNGhhqIz5K",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=" Paris",
                meta={
                    "content_index": 0,
                    "delta": " Paris",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 12,
                    "type": "response.output_text.delta",
                    "obfuscation": "ojbz89bS7j",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 13,
                    "text": "Germany: Berlin\nFrance: Paris",
                    "type": "response.output_text.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "output_index": 1,
                    "part": {
                        "annotations": [],
                        "text": "Germany: Berlin\nFrance: Paris",
                        "type": "output_text",
                        "logprobs": [],
                    },
                    "sequence_number": 14,
                    "type": "response.content_part.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                        "content": [
                            {
                                "annotations": [],
                                "text": "Germany: Berlin\nFrance: Paris",
                                "type": "output_text",
                                "logprobs": [],
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    },
                    "output_index": 1,
                    "sequence_number": 15,
                    "type": "response.output_item.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [
                            {
                                "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                                "summary": [],
                                "type": "reasoning",
                            },
                            {
                                "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                                "content": [
                                    {
                                        "annotations": [],
                                        "text": "Germany: Berlin\nFrance: Paris",
                                        "type": "output_text",
                                        "logprobs": [],
                                    }
                                ],
                                "role": "assistant",
                                "status": "completed",
                                "type": "message",
                            },
                        ],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "safety_identifier": None,
                        "service_tier": "default",
                        "status": "completed",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 15,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 77,
                            "output_tokens_details": {"reasoning_tokens": 64},
                            "total_tokens": 92,
                        },
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 16,
                    "type": "response.completed",
                },
                finish_reason="stop",
            ),
        ]

    def test_convert_only_function_call(self):
        chunks = [
            ResponseCreatedEvent(
                response=Response(
                    id="resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    created_at=1761907188.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[
                        FunctionTool(
                            name="weather",
                            parameters={
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                            strict=False,
                            type="function",
                            description="useful to determine the weather in a given location",
                        )
                    ],
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    usage=None,
                ),
                sequence_number=0,
                type="response.created",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseReasoningItem(
                    id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=2,
                type="response.output_item.added",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=3,
                type="response.output_item.done",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseFunctionToolCall(
                    arguments="",
                    call_id="call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                    name="weather",
                    type="function_call",
                    id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    status="in_progress",
                ),
                output_index=1,
                sequence_number=4,
                type="response.output_item.added",
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                delta='{"city":',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                output_index=1,
                sequence_number=5,
                type="response.function_call_arguments.delta",
                obfuscation="PySUcQ59ZZRkOm",
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                delta='"Paris"}',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                output_index=1,
                sequence_number=8,
                type="response.function_call_arguments.delta",
                obfuscation="INeMDAi1uAj",
            ),
            ResponseFunctionCallArgumentsDoneEvent(
                arguments='{"city":"Paris"}',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                name="weather",  # added name here because pydantic complains otherwise API returns a none here
                output_index=1,
                sequence_number=10,
                type="response.function_call_arguments.done",
            ),
            ResponseCompletedEvent(
                response=Response(
                    id="resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    created_at=1761907188.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[
                        ResponseReasoningItem(
                            id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                        ),
                        ResponseFunctionToolCall(
                            arguments='{"city":"Paris"}',
                            call_id="call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                            name="weather",
                            type="function_call",
                            id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            status="completed",
                        ),
                    ],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[
                        FunctionTool(
                            name="weather",
                            parameters={
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                            strict=False,
                            type="function",
                            description="useful to determine the weather in a given location",
                        )
                    ],
                    top_p=1.0,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    usage=ResponseUsage(
                        input_tokens=62,
                        input_tokens_details=InputTokensDetails(cached_tokens=0),
                        output_tokens=83,
                        output_tokens_details=OutputTokensDetails(reasoning_tokens=64),
                        total_tokens=145,
                    ),
                    store=True,
                ),
                sequence_number=12,
                type="response.completed",
            ),
        ]

        streaming_chunks = []
        for chunk in chunks:
            streaming_chunk = _convert_response_chunk_to_streaming_chunk(chunk, previous_chunks=streaming_chunks)
            streaming_chunks.append(streaming_chunk)

        assert streaming_chunks == [
            # TODO Unneeded streaming chunk
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                        "created_at": 1761907188.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [
                            {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                                "strict": False,
                                "type": "function",
                                "description": "useful to determine the weather in a given location",
                            }
                        ],
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "usage": None,
                    },
                    "sequence_number": 0,
                    "type": "response.created",
                },
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=0,
                start=True,
                reasoning=ReasoningContent(
                    reasoning_text="",
                    extra={
                        "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                        "summary": [],
                        "type": "reasoning",
                    },
                ),
            ),
            StreamingChunk(
                content="",
                meta={
                    "item": {
                        "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                        "summary": [],
                        "type": "reasoning",
                    },
                    "output_index": 0,
                    "sequence_number": 3,
                    "type": "response.output_item.done",
                    "received_at": ANY,
                },
                index=0,
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name="weather",
                        arguments=None,
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "arguments": "",
                            "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                            "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "name": "weather",
                            "status": "in_progress",
                            "type": "function_call",
                        },
                    )
                ],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name=None,
                        arguments='{"city":',
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "output_index": 1,
                            "sequence_number": 5,
                            "type": "response.function_call_arguments.delta",
                            "obfuscation": "PySUcQ59ZZRkOm",
                        },
                    )
                ],
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name=None,
                        arguments='"Paris"}',
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "output_index": 1,
                            "sequence_number": 8,
                            "type": "response.function_call_arguments.delta",
                            "obfuscation": "INeMDAi1uAj",
                        },
                    )
                ],
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "arguments": '{"city":"Paris"}',
                    "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    "name": "weather",
                    "output_index": 1,
                    "sequence_number": 10,
                    "type": "response.function_call_arguments.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                        "created_at": 1761907188.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [
                            {
                                "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                                "summary": [],
                                "type": "reasoning",
                            },
                            {
                                "arguments": '{"city":"Paris"}',
                                "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                                "name": "weather",
                                "type": "function_call",
                                "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                                "status": "completed",
                            },
                        ],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [
                            {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                                "strict": False,
                                "type": "function",
                                "description": "useful to determine the weather in a given location",
                            }
                        ],
                        "top_p": 1.0,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "usage": {
                            "input_tokens": 62,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 83,
                            "output_tokens_details": {"reasoning_tokens": 64},
                            "total_tokens": 145,
                        },
                        "store": True,
                    },
                    "sequence_number": 12,
                    "type": "response.completed",
                },
                finish_reason="tool_calls",
            ),
        ]
