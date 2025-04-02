from typing import List  # Import for explicit conversion

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, Toolset


# Define a simple math function
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers
    """
    return a + b


# Create a tool with proper schema
add_tool = Tool(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
    function=add_numbers,
)

# Create a toolset with the math tool
math_toolset = Toolset([add_tool])

# Create a complete pipeline that can use the toolset
pipeline = Pipeline()
pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo", tools=math_toolset))
pipeline.add_component("tool_invoker", ToolInvoker(tools=math_toolset))
pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_messages + tool_messages }}", output_type=list[ChatMessage], unsafe=True
    ),
)
pipeline.add_component("response_llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))

# Connect the components
pipeline.connect("llm.replies", "tool_invoker.messages")
pipeline.connect("llm.replies", "adapter.initial_tool_messages")
pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
pipeline.connect("adapter.output", "response_llm.messages")

# Use the pipeline with the toolset
user_input = "What is 5 plus 3?"
user_input_msg = ChatMessage.from_user(text=user_input)
result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})
# Result will contain "8" in the response
print(result)
