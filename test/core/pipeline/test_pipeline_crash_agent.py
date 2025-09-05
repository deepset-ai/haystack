# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
import sys

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, PipelineSnapshot, ToolBreakpoint
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


def run_complete_pipeline():
    pipeline, doc_store = build_pipeline()
    test_data = {
        "math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}
    }
    result = pipeline.run(data=test_data)
    print(result)
    print("Pipeline run completed. Stored documents:")
    for doc in doc_store.filter_documents():
        print(doc.content)


def run_pipeline_with_breakpoint_agent_llm():
    pipeline, doc_store = build_pipeline()
    test_data = {
        "math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}
    }
    agent_generator_breakpoint = Breakpoint(
        component_name="chat_generator", visit_count=0, snapshot_file_path="snapshots/"
    )
    agent_breakpoint = AgentBreakpoint(break_point=agent_generator_breakpoint, agent_name="math_agent")
    _ = pipeline.run(data=test_data, break_point=agent_breakpoint)


def resume_pipeline_from_snapshot(snapshot_file_path: str):
    pipeline, doc_store = build_pipeline()
    _ = {"math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}}
    # ToDo:
    #
    # "streaming_callback": {
    #     "type": "builtins.function"
    # }
    #
    # haystack.core.errors.PipelineRuntimeError: The following component failed to run:
    # Component name: 'math_agent'
    # Component type: 'Agent'
    # Error: Could not import class 'builtins.function'

    with open(snapshot_file_path) as f_in:
        raw_snapshot = json.load(f_in)
    snapshot = PipelineSnapshot.from_dict(raw_snapshot)
    result = pipeline.run(data={}, pipeline_snapshot=snapshot)
    print(result)
    print("Pipeline run completed. Stored documents:")
    for doc in doc_store.filter_documents():
        print(doc.content)


def run_pipeline_with_breakpoint_agent_tool():
    pipeline, doc_store = build_pipeline()
    test_data = {
        "math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}
    }
    agent_tool_breakpoint = ToolBreakpoint(
        tool_name="calculator", component_name="tool_invoker", visit_count=0, snapshot_file_path="snapshots/"
    )
    agent_breakpoint = AgentBreakpoint(break_point=agent_tool_breakpoint, agent_name="math_agent")
    _ = pipeline.run(data=test_data, break_point=agent_breakpoint)


def main():
    # run_complete_pipeline()
    # run_pipeline_with_breakpoint_agent_llm()
    # run_pipeline_with_breakpoint_agent_tool()
    resume_pipeline_from_snapshot(sys.argv[1])


if __name__ == "__main__":
    main()
