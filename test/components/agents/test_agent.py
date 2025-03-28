# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Iterator

from unittest.mock import MagicMock, patch
import pytest

from openai import Stream
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk

from haystack.components.agents import Agent
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.streaming_chunk import StreamingChunk
from haystack.tools import Tool, ComponentTool
from haystack.utils import serialize_callable, Secret

import os


def streaming_callback_for_serde(chunk: StreamingChunk):
    pass


def weather_function(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


weather_parameters = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters=weather_parameters,
        function=weather_function,
    )


@pytest.fixture
def component_tool():
    return ComponentTool(name="parrot", description="This is a parrot.", component=PromptBuilder(template="{{parrot}}"))


class OpenAIMockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)
        self.mock_chunk = mock_chunk

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        yield self.mock_chunk


@pytest.fixture
def openai_mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage=None,
        )
        mock_chat_completion_create.return_value = OpenAIMockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


class TestAgent:
    def test_serde(self, weather_tool, component_tool):
        os.environ["FAKE_OPENAI_KEY"] = "fake-key"
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(chat_generator=generator, tools=[weather_tool, component_tool])

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]

        assert serialized_agent["type"] == "haystack.components.agents.agent.Agent"
        assert (
            init_parameters["chat_generator"]["type"]
            == "haystack.components.generators.chat.openai.OpenAIChatGenerator"
        )
        assert init_parameters["streaming_callback"] == None
        assert init_parameters["tools"][0]["data"]["function"] == serialize_callable(weather_function)
        assert (
            init_parameters["tools"][1]["data"]["component"]["type"]
            == "haystack.components.builders.prompt_builder.PromptBuilder"
        )

        deserialized_agent = Agent.from_dict(serialized_agent)

        assert isinstance(deserialized_agent, Agent)
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert deserialized_agent.tools[0].function is weather_function
        assert isinstance(deserialized_agent.tools[1]._component, PromptBuilder)

    def test_serde_with_streaming_callback(self, weather_tool, component_tool):
        os.environ["FAKE_OPENAI_KEY"] = "fake-key"
        generator = OpenAIChatGenerator(api_key=Secret.from_env_var("FAKE_OPENAI_KEY"))
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
            streaming_callback=streaming_callback_for_serde,
        )

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]
        assert init_parameters["streaming_callback"] == "test_agent.streaming_callback_for_serde"

        deserialized_agent = Agent.from_dict(serialized_agent)
        assert deserialized_agent.streaming_callback is streaming_callback_for_serde

    def test_run_with_params_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        agent = Agent(chat_generator=chat_generator, streaming_callback=streaming_callback, tools=[weather_tool])
        agent.warm_up()
        response = agent.run([ChatMessage.from_user("Hello")])

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk

    def test_run_with_run_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        chat_generator = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))

        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        agent.warm_up()
        response = agent.run([ChatMessage.from_user("Hello")], streaming_callback=streaming_callback)

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk

    def test_keep_generator_streaming(self, openai_mock_chat_completion_chunk, weather_tool):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        chat_generator = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"), streaming_callback=streaming_callback
        )

        agent = Agent(chat_generator=chat_generator, tools=[weather_tool])
        agent.warm_up()
        response = agent.run([ChatMessage.from_user("Hello")])

        # check we called the streaming callback
        assert streaming_callback_called is True

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "messages" in response
        assert isinstance(response["messages"], list)
        assert len(response["messages"]) == 2
        assert [isinstance(reply, ChatMessage) for reply in response["messages"]]
        assert "Hello" in response["messages"][1].text  # see openai_mock_chat_completion_chunk
