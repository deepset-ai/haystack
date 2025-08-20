import math

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool


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


def main():
    snapshots_dir = "snapshots_agent_multiple_tool"
    agent = build_agent()
    agent.warm_up()
    response = agent.run(
        messages=[ChatMessage.from_user("What is 7 * (4 + 2)? What is the factorial of 5?")],
        state_persistence=True,
        state_persistence_path=snapshots_dir,
    )
    print(response["messages"])


if __name__ == "__main__":
    main()
