# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, Union
from unittest.mock import ANY, MagicMock, patch

import pytest
from openai import OpenAIError
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_message_function_tool_call import Function
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from openai.types.responses import (
    ParsedResponse,
    ParsedResponseOutputMessage,
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseUsage,
)
from pydantic import BaseModel

from haystack import component
from haystack.components.generators.chat.openai_responses import (
    OpenAIResponsesChatGenerator,
    _convert_response_to_chat_message,
    _convert_streaming_response_chunk_to_streaming_chunk,
)
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, print_streaming_chunk
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FinishReason,
    ImageContent,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCall,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import (
    ComponentTool,
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.http_client import init_http_client

logger = logging.getLogger(__name__)


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

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            OpenAIResponsesChatGenerator(tools=duplicate_tools)

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
                    "text_format": {
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIResponsesChatGenerator()
        results = component.run(chat_messages)
        print("NEW RESULTS")
        print(results)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-5-mini" in message.meta["model"]
        assert message.meta["status"] == "completed"
        assert message.meta["usage"]["total_tokens"] > 0

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
        assert message.reasoning is not None
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
        print(results)
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
        pass

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_and_streaming(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October12th at the Hilton Hotel downtown.")
        ]
        component = OpenAIResponsesChatGenerator(generation_kwargs={"text_format": calendar_event_model})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        print(message)
        msg = json.loads(message.text)
        assert "Marketing Summit" in msg["event_name"]
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

        assert message.meta["status"] == "completed"

    def test_run_with_wrong_model(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("Invalid model name")

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
        print(metadata)
        assert "gpt-5-mini" in metadata["model"]
        assert metadata["status"] == "completed"

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
        assert message.meta["status"] == "completed"

    def test_openai_chat_generator_with_toolset_initialization(self, tools, monkeypatch):
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
        assert message.meta["finish_reason"] == "tool_calls"

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

        generator = OpenAIResponsesChatGenerator(model="gpt-4.1-nano")
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.text
        assert "apple" in message.text.lower()

        assert message.is_from(ChatRole.ASSISTANT)
        assert not message.tool_calls
        assert not message.tool_call_results
