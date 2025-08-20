import math

from haystack import Document, Pipeline, component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools.tool import Tool

doc_store = InMemoryDocumentStore()


def calculate(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def factorial(n: int) -> dict:
    try:
        result = math.factorial(n)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def build_agent():
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
        chat_generator=OpenAIChatGenerator(),
        tools=[calculator_tool, factorial_tool],
        exit_conditions=["calculator"],
        streaming_callback=print_streaming_chunk,
        state_schema={"calc_result": {"type": int}, "factorial_result": {"type": int}},
    )

    return agent


@component
class ExtractResults:
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
    doc_writer = DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.SKIP)
    agent = build_agent()
    extractor = ExtractResults()

    pipe = Pipeline()
    pipe.add_component(instance=agent, name="math_agent")
    pipe.add_component(instance=extractor, name="extractor")
    pipe.add_component(instance=doc_writer, name="doc_writer")
    pipe.connect("math_agent.messages", "extractor.responses")
    pipe.connect("extractor.documents", "doc_writer.documents")

    return pipe


def main():
    snapshots_dir = "snapshots_agent_multiple_tool_in_pipeline"
    """
    agent = build_agent()
    agent.warm_up()
    response = agent.run(
        messages=[ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")],
        state_persistence=True,
        state_persistence_path=snapshots_dir,
    )
    print(response["messages"])
    """

    pipe = build_pipeline()
    pipe.run(
        data={"math_agent": {"messages": [ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")]}},
        state_persistence=True,
        state_persistence_path=snapshots_dir,
    )

    print("\nDocuments in store after run:\n")
    for doc in doc_store.filter_documents():
        print(doc)


if __name__ == "__main__":
    main()
