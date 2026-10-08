# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
from unittest.mock import Mock, patch

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import ChatMessage
from haystack.telemetry._telemetry import pipeline_running, tutorial_running
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret, TokenSecret


@patch("haystack.telemetry._telemetry.telemetry")
def test_pipeline_running(telemetry):
    telemetry.send_event = Mock()

    @component
    class Component:
        def _get_telemetry_data(self):
            return {"key": "values"}

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("component", Component())
    pipeline_running(pipe)

    expected_type = generate_qualified_class_name(type(pipe))
    # First run is always sent
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (3.x)",
        {
            "pipeline_id": str(id(pipe)),
            "pipeline_type": expected_type,
            "runs": 1,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )

    # Running again before one minute has passed should not send another event
    telemetry.send_event.reset_mock()
    pipeline_running(pipe)
    telemetry.send_event.assert_not_called()

    # Set the last telemetry sent time to pretend one minute has passed
    assert pipe._last_telemetry_sent is not None
    pipe._last_telemetry_sent = pipe._last_telemetry_sent - datetime.timedelta(minutes=1)

    telemetry.send_event.reset_mock()
    pipeline_running(pipe)
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (3.x)",
        {
            "pipeline_id": str(id(pipe)),
            "pipeline_type": expected_type,
            "runs": 3,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )

    # More than a day has passed but the seconds component of the timedelta is below the threshold:
    # the event must still be sent (regression test for using timedelta.seconds instead of total_seconds())
    pipe._last_telemetry_sent = datetime.datetime.now() - datetime.timedelta(days=1, seconds=5)

    telemetry.send_event.reset_mock()
    pipeline_running(pipe)
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (3.x)",
        {
            "pipeline_id": str(id(pipe)),
            "pipeline_type": expected_type,
            "runs": 4,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )


@patch("haystack.telemetry._telemetry.telemetry")
def test_pipeline_running_with_non_serializable_component(telemetry):
    telemetry.send_event = Mock()

    @component
    class Component:
        def __init__(self, api_key: Secret = TokenSecret("api_key")):
            self.api_key = api_key

        def _get_telemetry_data(self):
            return {"key": "values"}

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("component", Component())
    pipeline_running(pipe)
    telemetry.send_event.assert_called_once_with(
        "Pipeline run (3.x)",
        {
            "pipeline_id": str(id(pipe)),
            "pipeline_type": "haystack.core.pipeline.pipeline.Pipeline",
            "runs": 1,
            "components": {"test.test_telemetry.Component": [{"name": "component", "key": "values"}]},
        },
    )


@patch("haystack.telemetry._telemetry.telemetry")
def test_pipeline_running_reports_agent_tools(telemetry):
    telemetry.send_event = Mock()
    tool = Tool(name="tool", description="A tool.", parameters={"type": "object", "properties": {}}, function=print)

    pipe = Pipeline()
    pipe.add_component("agent", Agent(chat_generator=MockChatGenerator("Hello"), tools=[tool]))
    pipeline_running(pipe)

    event_properties = telemetry.send_event.call_args.args[1]
    assert event_properties["components"]["haystack.components.agents.agent.Agent"] == [
        {
            "name": "agent",
            "chat_generator": {"type": "haystack.components.generators.chat.mock.MockChatGenerator"},
            "tools": {"count": 1, "tools": [{"type": "haystack.tools.tool.Tool", "name": "tool"}], "toolset_types": {}},
        }
    ]


class _ToolsetLoadingOnWarmUp(Toolset):
    def __init__(self, tool: Tool) -> None:
        super().__init__(tools=[])
        self._tool = tool

    def warm_up(self) -> None:
        self.tools = [self._tool]


def _pipeline_with_agent_using_toolset_loading_on_warm_up() -> Pipeline:
    tool = Tool(name="tool", description="A tool.", parameters={"type": "object", "properties": {}}, function=print)
    pipe = Pipeline()
    pipe.add_component(
        "agent", Agent(chat_generator=MockChatGenerator("Hello"), tools=_ToolsetLoadingOnWarmUp(tool=tool))
    )
    return pipe


def _reported_agent_tools(capture: Mock) -> dict:
    return capture.call_args.kwargs["properties"]["components"]["haystack.components.agents.agent.Agent"][0]["tools"]


def test_pipeline_run_reports_tools_loaded_on_warm_up(block_telemetry_network_calls: Mock) -> None:
    pipe = _pipeline_with_agent_using_toolset_loading_on_warm_up()

    pipe.run({"agent": {"messages": [ChatMessage.from_user("Hi")]}})

    assert _reported_agent_tools(block_telemetry_network_calls)["tools"] == [
        {"type": "haystack.tools.tool.Tool", "name": "tool"}
    ]


async def test_pipeline_run_async_reports_tools_loaded_on_warm_up(block_telemetry_network_calls: Mock) -> None:
    pipe = _pipeline_with_agent_using_toolset_loading_on_warm_up()

    await pipe.run_async({"agent": {"messages": [ChatMessage.from_user("Hi")]}})

    assert _reported_agent_tools(block_telemetry_network_calls)["tools"] == [
        {"type": "haystack.tools.tool.Tool", "name": "tool"}
    ]


def test_pipeline_running_with_non_dict_telemetry_data(caplog):
    @component
    class Component:
        def __init__(self, api_key: Secret = TokenSecret("api_key")):
            self.api_key = api_key

        # telemetry data should be a dictionary but is a list
        def _get_telemetry_data(self):
            return ["values"]

        @component.output_types(value=int)
        def run(self):
            pass

    pipe = Pipeline()
    pipe.add_component("my_component", Component())
    with caplog.at_level(logging.DEBUG):
        pipeline_running(pipe)
        assert "TypeError: Telemetry data for component my_component must be a dictionary" in caplog.text


def test_send_telemetry_preserves_function_metadata():
    """
    Regression test for https://github.com/deepset-ai/haystack/issues/11568.

    The ``send_telemetry`` decorator must use ``functools.wraps`` so that decorated functions such as
    ``pipeline_running`` and ``tutorial_running`` keep their own metadata (``__name__``, ``__doc__`` and
    ``__annotations__``) instead of exposing the ``send_telemetry_wrapper`` wrapper's.
    """
    # ``__name__`` comes from the wrapped function, not the wrapper.
    assert pipeline_running.__name__ == "pipeline_running"
    assert tutorial_running.__name__ == "tutorial_running"

    # ``__doc__`` is preserved.
    assert pipeline_running.__doc__ is not None
    assert "Collects telemetry data for a pipeline run" in pipeline_running.__doc__

    # ``__annotations__`` are preserved, e.g. the wrapped functions' parameters.
    assert "pipeline" in pipeline_running.__annotations__
    assert "tutorial_id" in tutorial_running.__annotations__

    # ``functools.wraps`` also exposes the undecorated function through ``__wrapped__``.
    assert hasattr(pipeline_running, "__wrapped__")
    assert pipeline_running.__wrapped__.__name__ == "pipeline_running"
