# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Annotated

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.core.errors import BreakpointException
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool, create_tool_from_function


def calculate(expression: Annotated[str, "Math expression to evaluate"]) -> dict:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def factorial(n: Annotated[int, "Number to calculate the factorial of"]) -> dict:
    """Calculate the factorial of a number."""
    try:
        result = math.factorial(n)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@component
class FakeChatGenerator:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.count = 0

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: list[Tool] = None, **kwargs):
        if self.count >= len(self.responses):
            return {"replies": [ChatMessage.from_assistant("Final answer")]}
        res = self.responses[self.count]
        self.count += 1
        return res


def build_agent(chat_generator):
    """Build an agent with calculator and factorial tools."""
    factorial_tool = create_tool_from_function(
        function=factorial, outputs_to_state={"factorial_result": {"source": "result"}}
    )

    calculator_tool = create_tool_from_function(
        function=calculate, name="calculator", outputs_to_state={"calc_result": {"source": "result"}}
    )

    agent = Agent(
        chat_generator=chat_generator,
        tools=[calculator_tool, factorial_tool],
        exit_conditions=["calculator"],
        streaming_callback=print_streaming_chunk,
        state_schema={"calc_result": {"type": int}, "factorial_result": {"type": int}},
    )

    return agent


@component
class ExtractResults:
    """Component to extract results from agent responses."""

    @component.output_types(documents=list[Document])
    def run(self, responses: list[ChatMessage]) -> dict:
        results = []
        for msg in responses:
            if msg.text:
                results.append(Document(content=msg.text))
            elif isinstance(msg._content[0], ToolCall):
                results.extend(Document(content=f"{tc.tool_name} - Arguments: {tc.arguments}") for tc in msg._content)
            elif isinstance(msg._content[0], ToolCallResult):
                results.extend(Document(content=f"{tr.origin.tool_name} - Result: {tr.result}") for tr in msg._content)
        return {"documents": results}


class TestPipelineBreakpoints:
    @pytest.fixture
    def agent_pipeline(self):
        """Create a pipeline with agent, extractor, and document writer for testing."""
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.OVERWRITE)

        mock_responses = [
            {
                "replies": [
                    ChatMessage.from_assistant(
                        None,
                        tool_calls=[
                            ToolCall(tool_name="calculator", arguments={"expression": "2+2"}),
                            ToolCall(tool_name="factorial", arguments={"n": 5}),
                        ],
                    )
                ]
            },
            {"replies": [ChatMessage.from_assistant("The result of 2*2 is 4. The factorial of 5 is 120.")]},
        ]

        agent = build_agent(FakeChatGenerator(responses=mock_responses))
        extractor = ExtractResults()

        pipe = Pipeline()
        pipe.add_component(instance=agent, name="math_agent")
        pipe.add_component(instance=extractor, name="extractor")
        pipe.add_component(instance=doc_writer, name="doc_writer")
        pipe.connect("math_agent.messages", "extractor.responses")
        pipe.connect("extractor.documents", "doc_writer.documents")

        return pipe, doc_store

    BREAKPOINT_COMPONENTS = ["math_agent", "extractor", "doc_writer"]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_agent_pipeline_regular_component_breakpoints(
        self, agent_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
        pipeline, doc_store = agent_pipeline
        data = {
            "math_agent": {
                "messages": [
                    ChatMessage.from_user(
                        "Use the calculator tool to calculate 2 + 2, factorial tool for the factorial of 5."
                    )
                ]
            }
        }

        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=pipeline, output_directory=output_directory, component_name=break_point.component_name, data=data
        )
        assert result["math_agent"]["calc_result"] == 4
        assert result["math_agent"]["factorial_result"] == 120
        assert result["doc_writer"]["documents_written"] == 5

    @pytest.fixture
    def agent_breakpoints(self, output_directory):
        common = {"visit_count": 0, "snapshot_file_path": str(output_directory)}
        return [
            AgentBreakpoint(agent_name="math_agent", break_point=Breakpoint(component_name="chat_generator", **common)),
            AgentBreakpoint(
                agent_name="math_agent",
                break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="calculator", **common),
            ),
            AgentBreakpoint(
                agent_name="math_agent",
                break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="factorial", **common),
            ),
        ]

    @pytest.mark.parametrize("breakpoint_index", [0, 1, 2], ids=["chat_generator", "calculator_tool", "factorial_tool"])
    @pytest.mark.integration
    def test_agent_components_pipeline_breakpoints(
        self, agent_breakpoints, agent_pipeline, output_directory, breakpoint_index, load_and_resume_pipeline_snapshot
    ):
        pipeline, doc_store = agent_pipeline
        data = {
            "math_agent": {
                "messages": [
                    ChatMessage.from_user(
                        "Use the calculator tool to calculate 7 * (4 + 2), factorial tool for the factorial of 5."
                    )
                ]
            }
        }

        # Get the specific breakpoint from the fixture list
        agent_breakpoint = agent_breakpoints[breakpoint_index]

        try:
            _ = pipeline.run(data, break_point=agent_breakpoint)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=pipeline,
            output_directory=output_directory,
            component_name="math_agent_" + agent_breakpoint.break_point.component_name,
            data=data,
        )

        assert result["math_agent"]["calc_result"] == 4
        assert result["math_agent"]["factorial_result"] == 120
        assert result["doc_writer"]["documents_written"] != 0
