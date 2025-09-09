# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.tool import Tool


def calculate(expression: str) -> dict:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def factorial(n: int) -> dict:
    raise Exception("Error in factorial tool")  # Simulate a crash in the tool


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
        raise_on_tool_invocation_failure=True,
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


def build_pipeline():
    """Build a pipeline with agent, extractor, and document writer."""
    doc_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP)
    agent = build_agent()
    extractor = ExtractResults()

    pipe = Pipeline()
    pipe.add_component(instance=agent, name="math_agent")
    pipe.add_component(instance=extractor, name="extractor")
    pipe.add_component(instance=doc_writer, name="doc_writer")
    pipe.connect("math_agent.messages", "extractor.responses")
    pipe.connect("extractor.documents", "doc_writer.documents")

    return pipe, doc_store


@pytest.mark.integration
def test_pipeline_with_include_outputs_from():
    pipe, doc_store = build_pipeline()

    test_data = {
        "math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}
    }

    with pytest.raises(PipelineRuntimeError) as exception_info:
        _ = pipe.run(data=test_data)

    pipeline_snapshot = exception_info.value.pipeline_snapshot
    assert pipeline_snapshot is not None, "Pipeline snapshot should be captured in the exception"
    assert "Error in factorial tool" in str(exception_info.value), "Exception message should contain tool error"

    assert pipeline_snapshot.original_input_data is not None
    assert pipeline_snapshot.ordered_component_names == ["doc_writer", "extractor", "math_agent"]
    assert pipeline_snapshot.pipeline_state.component_visits == {"doc_writer": 0, "extractor": 0, "math_agent": 0}

    # AgentBreakpoint is correctly set
    assert pipeline_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.break_point.break_point.component_name == "tool_invoker"
    assert pipeline_snapshot.break_point.break_point.visit_count == 0
    assert pipeline_snapshot.break_point.break_point.tool_name == "calculator"

    # AgentSnapshot is correctly set
    assert pipeline_snapshot.agent_snapshot.component_inputs.keys() == {"chat_generator", "tool_invoker"}
    assert pipeline_snapshot.agent_snapshot.component_visits == {"chat_generator": 1, "tool_invoker": 0}
    assert pipeline_snapshot.agent_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.component_name == "tool_invoker"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.visit_count == 0
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.tool_name == "calculator"

    # Test if we can resume the pipeline from the generated snapshot
    _ = pipe.run(data={}, pipeline_snapshot=pipeline_snapshot)
