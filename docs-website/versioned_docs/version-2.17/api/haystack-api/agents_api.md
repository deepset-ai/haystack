---
title: Agents
id: agents-api
description: Tool-using agents with provider-agnostic chat model support.
---

<a id="agent"></a>

# Module agent

<a id="agent.Agent"></a>

## Agent

A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

The component processes messages and executes tools until an exit_condition condition is met.
The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

### Usage example
```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool

# Define tools with all required parameters
def calculate(operation: str, a: float, b: float) -> float:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0

def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

tools = [
    Tool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculate
    ),
    Tool(
        name="search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        },
        function=search
    )
]

# Note: OpenAIChatGenerator requires OPENAI_API_KEY environment variable to be set
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=tools,
    exit_conditions=["search"],
)

# Run the agent
result = agent.run(
    messages=[ChatMessage.from_user("Find information about Haystack")]
)

assert "messages" in result  # Contains conversation history
```

<a id="agent.Agent.__init__"></a>

#### Agent.\_\_init\_\_

```python
def __init__(*,
             chat_generator: ChatGenerator,
             tools: Optional[Union[list[Tool], Toolset]] = None,
             system_prompt: Optional[str] = None,
             exit_conditions: Optional[list[str]] = None,
             state_schema: Optional[dict[str, Any]] = None,
             max_agent_steps: int = 100,
             streaming_callback: Optional[StreamingCallbackT] = None,
             raise_on_tool_invocation_failure: bool = False,
             tool_invoker_kwargs: Optional[dict[str, Any]] = None) -> None
```

Initialize the agent component.

**Arguments**:

- `chat_generator`: An instance of the chat generator that your agent should use. It must support tools.
- `tools`: List of Tool objects or a Toolset that the agent can use.
- `system_prompt`: System prompt for the agent.
- `exit_conditions`: List of conditions that will cause the agent to return.
Can include "text" if the agent should return when it generates a message without tool calls,
or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
- `state_schema`: The schema for the runtime state used by the tools.
- `max_agent_steps`: Maximum number of steps the agent will run before stopping. Defaults to 100.
If the agent exceeds this number of steps, it will stop and return the current state.
- `streaming_callback`: A callback that will be invoked when a response is streamed from the LLM.
The same callback can be configured to emit tool results when a tool is called.
- `raise_on_tool_invocation_failure`: Should the agent raise an exception when a tool invocation fails?
If set to False, the exception will be turned into a chat message and passed to the LLM.
- `tool_invoker_kwargs`: Additional keyword arguments to pass to the ToolInvoker.

**Raises**:

- `TypeError`: If the chat_generator does not support tools parameter in its run method.
- `ValueError`: If the exit_conditions are not valid.

<a id="agent.Agent.warm_up"></a>

#### Agent.warm\_up

```python
def warm_up() -> None
```

Warm up the Agent.

<a id="agent.Agent.to_dict"></a>

#### Agent.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data

<a id="agent.Agent.from_dict"></a>

#### Agent.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Agent"
```

Deserialize the agent from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from

**Returns**:

Deserialized agent

<a id="agent.Agent.run"></a>

#### Agent.run

```python
def run(messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        **kwargs: Any) -> dict[str, Any]
```

Process messages and execute tools until an exit condition is met.

**Arguments**:

- `messages`: List of Haystack ChatMessage objects to process.
If a list of dictionaries is provided, each dictionary will be converted to a ChatMessage object.
- `streaming_callback`: A callback that will be invoked when a response is streamed from the LLM.
The same callback can be configured to emit tool results when a tool is called.
- `break_point`: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
for "tool_invoker".
- `snapshot`: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
the relevant information to restart the Agent execution from where it left off.
- `kwargs`: Additional data to pass to the State schema used by the Agent.
The keys must match the schema defined in the Agent's `state_schema`.

**Raises**:

- `RuntimeError`: If the Agent component wasn't warmed up before calling `run()`.
- `BreakpointException`: If an agent breakpoint is triggered.

**Returns**:

A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.

<a id="agent.Agent.run_async"></a>

#### Agent.run\_async

```python
async def run_async(messages: list[ChatMessage],
                    streaming_callback: Optional[StreamingCallbackT] = None,
                    *,
                    break_point: Optional[AgentBreakpoint] = None,
                    snapshot: Optional[AgentSnapshot] = None,
                    **kwargs: Any) -> dict[str, Any]
```

Asynchronously process messages and execute tools until the exit condition is met.

This is the asynchronous version of the `run` method. It follows the same logic but uses
asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
if available.

**Arguments**:

- `messages`: List of chat messages to process
- `streaming_callback`: An asynchronous callback that will be invoked when a response
is streamed from the LLM. The same callback can be configured to emit tool results when a tool is called.
- `break_point`: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
for "tool_invoker".
- `snapshot`: A dictionary containing the state of a previously saved agent execution.
- `kwargs`: Additional data to pass to the State schema used by the Agent.
The keys must match the schema defined in the Agent's `state_schema`.

**Raises**:

- `RuntimeError`: If the Agent component wasn't warmed up before calling `run_async()`.
- `BreakpointException`: If an agent breakpoint is triggered.

**Returns**:

A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.
