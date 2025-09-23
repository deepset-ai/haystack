# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.core.errors import BreakpointException
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.tool import Tool
from test.conftest import load_and_resume_pipeline_snapshot


def calculate(expression: str) -> dict:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def factorial(n: int) -> dict:
    """Calculate the factorial of a number."""
    try:
        result = math.factorial(n)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def build_agent():
    """Build an agent with calculator and factorial tools."""
    factorial_tool = Tool(
        name="factorial",
        description="Calculate the factorial of a number.",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number to calculate the factorial of"}},
            "required": ["n"],
        },
        function=factorial,
        outputs_to_state={"factorial_result": {"source": "result"}},
    )

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

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
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
            if text := msg.text:
                results.append(Document(content=f"{text}"))
                continue

            # If the message contains ToolCall object extract the tool name, arguments and arguments
            if isinstance(msg._content[0], ToolCall):
                for tool_call in msg._content:
                    tool_name = tool_call.tool_name
                    arguments = tool_call.arguments
                    results.append(Document(content=f"{tool_name} - Arguments: {arguments}"))

            # If the message contains ToolCallResult extract the tool name, arguments and arguments
            if isinstance(msg._content[0], ToolCallResult):
                for tool_call_result in msg._content:
                    tool_name = tool_call_result.origin.tool_name
                    result = tool_call_result.result
                    results.append(Document(content=f"{tool_name} - Result: {result}"))

        return {"documents": results}


class TestPipelineBreakpoints:
    @pytest.fixture
    def agent_pipeline(self):
        """Create a pipeline with agent, extractor, and document writer for testing."""
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.OVERWRITE)

        # Build agent with mocked chat generator
        factorial_tool = Tool(
            name="factorial",
            description="Calculate the factorial of a number.",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Number to calculate the factorial of"}},
                "required": ["n"],
            },
            function=factorial,
            outputs_to_state={"factorial_result": {"source": "result"}},
        )

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

        agent = Agent(
            chat_generator=OpenAIChatGenerator(),
            tools=[calculator_tool, factorial_tool],
            exit_conditions=["calculator"],
            streaming_callback=print_streaming_chunk,
            state_schema={"calc_result": {"type": int}, "factorial_result": {"type": int}},
        )

        extractor = ExtractResults()

        pipe = Pipeline()
        pipe.add_component(instance=agent, name="math_agent")
        pipe.add_component(instance=extractor, name="extractor")
        pipe.add_component(instance=doc_writer, name="doc_writer")
        pipe.connect("math_agent.messages", "extractor.responses")
        pipe.connect("extractor.documents", "doc_writer.documents")

        return pipe, doc_store

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory) -> Path:
        return tmp_path_factory.mktemp("output_files")

    BREAKPOINT_COMPONENTS = ["math_agent", "extractor", "doc_writer"]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_agent_pipeline_regular_component_breakpoints(self, agent_pipeline, output_directory, component):
        pipeline, doc_store = agent_pipeline
        data = {"math_agent": {"messages": [ChatMessage.from_user("Calculate 2 + 2. What is the factorial of 5?")]}}

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

    @pytest.fixture(scope="session")
    def agent_breakpoints(self, output_directory):
        return [
            # Chat Generator breakpoint
            AgentBreakpoint(
                break_point=Breakpoint(
                    component_name="chat_generator", visit_count=0, snapshot_file_path=str(output_directory)
                ),
                agent_name="math_agent",
            ),
            # Tool Call - Calculator
            AgentBreakpoint(
                break_point=ToolBreakpoint(
                    component_name="tool_invoker",
                    tool_name="calculator",
                    visit_count=0,
                    snapshot_file_path=str(output_directory),
                ),
                agent_name="math_agent",
            ),
            # Tool Call - Factorial
            AgentBreakpoint(
                break_point=ToolBreakpoint(
                    component_name="tool_invoker",
                    tool_name="factorial",
                    visit_count=0,
                    snapshot_file_path=str(output_directory),
                ),
                agent_name="math_agent",
            ),
        ]

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.parametrize("breakpoint_index", [0, 1, 2], ids=["chat_generator", "calculator_tool", "factorial_tool"])
    @pytest.mark.integration
    def test_agent_components_pipeline_breakpoints(
        self, agent_breakpoints, agent_pipeline, output_directory, breakpoint_index
    ):
        pipeline, doc_store = agent_pipeline
        data = {"math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}}

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

        assert result["math_agent"]["calc_result"] == 42
        assert result["math_agent"]["factorial_result"] == 120
        assert result["doc_writer"]["documents_written"] != 0
