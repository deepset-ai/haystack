# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.core.errors import BreakpointException
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.tool import Tool
from haystack.utils.auth import Secret
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
    def mock_openai_chat_generator(self):
        """
        Creates a mock for the OpenAIChatGenerator.
        """
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            # Create mock completion objects
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(
                        content="I'll help you calculate that. Let me use the calculator tool.",
                        tool_calls=[
                            MagicMock(
                                id="call_123",
                                type="function",
                                function=MagicMock(name="calculator", arguments='{"expression": "2 + 2"}'),
                            )
                        ],
                    ),
                )
            ]
            mock_completion.usage = {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97}

            mock_chat_completion_create.return_value = mock_completion

            # Create a mock for the OpenAIChatGenerator
            @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
            def create_mock_generator(model_name):
                generator = OpenAIChatGenerator(model=model_name, api_key=Secret.from_env_var("OPENAI_API_KEY"))

                # Mock the run method
                def mock_run(messages, streaming_callback=None, generation_kwargs=None, tools=None, tools_strict=None):
                    # Check if this is a tool call response
                    if any("tool_call" in str(msg) for msg in messages):
                        content = "The result is 4."
                    else:
                        content = "I'll help you calculate that. Let me use the calculator tool."

                    return {
                        "replies": [ChatMessage.from_assistant(content)],
                        "meta": {
                            "model": model_name,
                            "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
                        },
                    }

                # Replace the run method with our mock
                generator.run = mock_run

                return generator

            yield create_mock_generator

    @pytest.fixture
    def agent_pipeline(self, mock_openai_chat_generator):
        """Create a pipeline with agent, extractor, and document writer for testing."""
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP)

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
            chat_generator=mock_openai_chat_generator("gpt-4o-mini"),
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

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_agent_pipeline_breakpoints(self, agent_pipeline, output_directory, component):
        pipeline, doc_store = agent_pipeline
        data = {"math_agent": {"messages": [ChatMessage.from_user("Calculate 2 + 2")]}}

        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=pipeline, output_directory=output_directory, component_name=break_point.component_name, data=data
        )
        assert result["doc_writer"]
