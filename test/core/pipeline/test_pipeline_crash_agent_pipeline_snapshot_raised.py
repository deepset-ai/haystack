# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import pytest

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.tools import ToolInvoker
from haystack.components.writers import DocumentWriter
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import Tool, Toolset, create_tool_from_function


def calculate(expression: str) -> dict:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def factorial_failing(n: int) -> dict:
    """Calculate the factorial of a number."""
    raise Exception("Error in factorial tool")  # Simulate a crash in the tool


failing_factorial_tool = create_tool_from_function(
    function=factorial_failing, name="factorial", outputs_to_state={"factorial_result": {"source": "result"}}
)

calculator_tool = create_tool_from_function(
    function=calculate, name="calculator", outputs_to_state={"calc_result": {"source": "result"}}
)


@component
class MockChatGenerator:
    def __init__(self, fail_on_call: bool):
        self.fail_on_call = fail_on_call

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Optional[Union[list[Tool], Toolset]] = None, **kwargs) -> dict:
        if self.fail_on_call:
            # Simulate a crash in the chat generator
            raise Exception("Error in chat generator component")
        else:
            return {
                "replies": [
                    ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="factorial", arguments={"n": 5})])
                ]
            }


@component
class ExtractResults:
    @component.output_types(documents=list[Document])
    def run(self, responses: list[ChatMessage]) -> dict:
        return {"documents": [Document(content=resp.text) for resp in responses]}


def build_pipeline(agent: Agent):
    """Build a pipeline with the given agent, extractor, and document writer."""
    doc_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP)
    extractor = ExtractResults()

    pipe = Pipeline()
    pipe.add_component(instance=agent, name="math_agent")
    pipe.add_component(instance=extractor, name="extractor")
    pipe.add_component(instance=doc_writer, name="doc_writer")
    pipe.connect("math_agent.messages", "extractor.responses")
    pipe.connect("extractor.documents", "doc_writer.documents")

    return pipe


def test_pipeline_with_chat_generator_crash():
    """Test pipeline crash handling when chat generator fails."""
    pipe = build_pipeline(
        agent=Agent(
            chat_generator=MockChatGenerator(True), tools=[calculator_tool], state_schema={"calc_result": {"type": int}}
        )
    )

    test_data = {"math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)?")]}}

    with pytest.raises(PipelineRuntimeError) as exception_info:
        _ = pipe.run(data=test_data)

    assert "Error in chat generator component" in str(exception_info.value)
    assert exception_info.value.component_name == "chat_generator"
    assert exception_info.value.component_type == MockChatGenerator
    assert "math_agent_chat_generator" in exception_info.value.pipeline_snapshot_file_path

    pipeline_snapshot = exception_info.value.pipeline_snapshot
    assert pipeline_snapshot is not None, "Pipeline snapshot should be captured in the exception"

    assert pipeline_snapshot.original_input_data is not None
    assert pipeline_snapshot.ordered_component_names == ["doc_writer", "extractor", "math_agent"]
    assert pipeline_snapshot.pipeline_state.component_visits == {"doc_writer": 0, "extractor": 0, "math_agent": 0}

    # AgentBreakpoint is correctly set for chat_generator crash
    assert pipeline_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.break_point.break_point.component_name == "chat_generator"
    assert pipeline_snapshot.break_point.break_point.visit_count == 0

    # AgentSnapshot is correctly set
    assert pipeline_snapshot.agent_snapshot.component_inputs.keys() == {"chat_generator", "tool_invoker"}
    assert pipeline_snapshot.agent_snapshot.component_visits == {"chat_generator": 0, "tool_invoker": 0}
    assert pipeline_snapshot.agent_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.component_name == "chat_generator"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.visit_count == 0

    # Test if we can resume the pipeline from the generated snapshot. Note that, the pipeline should fail again with
    # the same error since we are resuming from the same exact state and did not change anything in the pipeline.
    #
    # We passed it to the pipeline.run() to make sure that the generated snapshot is valid and indeed the pipeline
    # resumes
    with pytest.raises(PipelineRuntimeError):
        _ = pipe.run(data={}, pipeline_snapshot=pipeline_snapshot)


def test_pipeline_with_tool_call_crash():
    """Test pipeline crash handling when a tool call fails."""
    pipe = build_pipeline(
        agent=Agent(
            chat_generator=MockChatGenerator(False),
            tools=[calculator_tool, failing_factorial_tool],
            state_schema={"calc_result": {"type": int}, "factorial_result": {"type": int}},
            raise_on_tool_invocation_failure=True,
        )
    )

    test_data = {
        "math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}
    }

    with pytest.raises(PipelineRuntimeError) as exception_info:
        _ = pipe.run(data=test_data)

    assert "Error in factorial tool" in str(exception_info.value), "Exception message should contain tool error"
    assert exception_info.value.component_name == "tool_invoker"
    assert exception_info.value.component_type == ToolInvoker
    assert "math_agent_tool_invoker" in exception_info.value.pipeline_snapshot_file_path

    pipeline_snapshot = exception_info.value.pipeline_snapshot
    assert pipeline_snapshot is not None, "Pipeline snapshot should be captured in the exception"

    assert pipeline_snapshot.original_input_data is not None
    assert pipeline_snapshot.ordered_component_names == ["doc_writer", "extractor", "math_agent"]
    assert pipeline_snapshot.pipeline_state.component_visits == {"doc_writer": 0, "extractor": 0, "math_agent": 0}

    # AgentBreakpoint is correctly set
    assert pipeline_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.break_point.break_point.component_name == "tool_invoker"
    assert pipeline_snapshot.break_point.break_point.visit_count == 0
    assert pipeline_snapshot.break_point.break_point.tool_name == "factorial"

    # AgentSnapshot is correctly set
    assert pipeline_snapshot.agent_snapshot.component_inputs.keys() == {"chat_generator", "tool_invoker"}
    assert pipeline_snapshot.agent_snapshot.component_visits == {"chat_generator": 1, "tool_invoker": 0}
    assert pipeline_snapshot.agent_snapshot.break_point.agent_name == "math_agent"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.component_name == "tool_invoker"
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.visit_count == 0
    assert pipeline_snapshot.agent_snapshot.break_point.break_point.tool_name == "factorial"

    # Test if the pipeline can be resumed from the generated snapshot. Note that, the pipeline should fail again with
    # the same error since we are resuming from the same exact state and did not change anything in the pipeline.
    with pytest.raises(PipelineRuntimeError):
        _ = pipe.run(data={}, pipeline_snapshot=pipeline_snapshot)
