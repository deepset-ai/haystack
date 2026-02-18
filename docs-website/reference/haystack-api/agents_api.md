---
title: "Agents"
id: agents-api
description: "Tool-using agents with provider-agnostic chat model support."
slug: "/agents-api"
---


## agent

### Agent

A tool-using Agent powered by a large language model.

The Agent processes messages and calls tools until it meets an exit condition.
You can set one or more exit conditions to control when it stops.
For example, it can stop after generating a response or after calling a tool.

Without tools, the Agent works like a standard LLM that generates text. It produces one response and then stops.

### Usage examples

This is an example agent that:

1. Searches for tipping customs in France.
1. Uses a calculator to compute tips based on its findings.
1. Returns the final answer with its context.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

# Tool functions - in practice, these would have real implementations
def search(query: str) -> str:
    '''Search for information on the web.'''
    # Placeholder: would call actual search API
    return "In France, a 15% service charge is typically included, but leaving 5-10% extra is appreciated."

def calculator(operation: str, a: float, b: float) -> float:
    '''Perform mathematical calculations.'''
    if operation == "multiply":
        return a * b
    elif operation == "percentage":
        return (a / 100) * b
    return 0

# Define tools with JSON Schema
tools = [
    Tool(
        name="search",
        description="Searches for information on the web",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        },
        function=search
    ),
    Tool(
        name="calculator",
        description="Performs mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "Operation: multiply, percentage"},
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculator
    )
]

# Create and run the agent
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=tools
)

result = agent.run(
    messages=[ChatMessage.from_user("Calculate the appropriate tip for an €85 meal in France")]
)

print(result["messages"][-1].text)
```

#### __init__

```python
__init__(
    *,
    chat_generator: ChatGenerator,
    tools: ToolsType | None = None,
    system_prompt: str | None = None,
    exit_conditions: list[str] | None = None,
    state_schema: dict[str, Any] | None = None,
    max_agent_steps: int = 100,
    streaming_callback: StreamingCallbackT | None = None,
    raise_on_tool_invocation_failure: bool = False,
    tool_invoker_kwargs: dict[str, Any] | None = None,
    confirmation_strategies: (
        dict[str | tuple[str, ...], ConfirmationStrategy] | None
    ) = None
) -> None
```

Initialize the agent component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator</code>) – An instance of the chat generator that your agent should use. It must support tools.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
- **system_prompt** (<code>str | None</code>) – System prompt for the agent.
- **exit_conditions** (<code>list\[str\] | None</code>) – List of conditions that will cause the agent to return.
  Can include "text" if the agent should return when it generates a message without tool calls,
  or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
- **state_schema** (<code>dict\[str, Any\] | None</code>) – The schema for the runtime state used by the tools.
- **max_agent_steps** (<code>int</code>) – Maximum number of steps the agent will run before stopping. Defaults to 100.
  If the agent exceeds this number of steps, it will stop and return the current state.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback that will be invoked when a response is streamed from the LLM.
  The same callback can be configured to emit tool results when a tool is called.
- **raise_on_tool_invocation_failure** (<code>bool</code>) – Should the agent raise an exception when a tool invocation fails?
  If set to False, the exception will be turned into a chat message and passed to the LLM.
- **tool_invoker_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments to pass to the ToolInvoker.
- **confirmation_strategies** (<code>dict\[str | tuple\[str, ...\], ConfirmationStrategy\] | None</code>) – A dictionary mapping tool names to ConfirmationStrategy instances.

**Raises:**

- <code>TypeError</code> – If the chat_generator does not support tools parameter in its run method.
- <code>ValueError</code> – If the exit_conditions are not valid.

#### warm_up

```python
warm_up() -> None
```

Warm up the Agent.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Agent
```

Deserialize the agent from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from

**Returns:**

- <code>Agent</code> – Deserialized agent

#### run

```python
run(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    *,
    generation_kwargs: dict[str, Any] | None = None,
    break_point: AgentBreakpoint | None = None,
    snapshot: AgentSnapshot | None = None,
    system_prompt: str | None = None,
    tools: ToolsType | list[str] | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None,
    **kwargs: Any
) -> dict[str, Any]
```

Process messages and execute tools until an exit condition is met.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – List of Haystack ChatMessage objects to process.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback that will be invoked when a response is streamed from the LLM.
  The same callback can be configured to emit tool results when a tool is called.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for LLM. These parameters will
  override the parameters passed during component initialization.
- **break_point** (<code>AgentBreakpoint | None</code>) – An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
  for "tool_invoker".
- **snapshot** (<code>AgentSnapshot | None</code>) – A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
  the relevant information to restart the Agent execution from where it left off.
- **system_prompt** (<code>str | None</code>) – System prompt for the agent. If provided, it overrides the default system prompt.
- **tools** (<code>ToolsType | list\[str\] | None</code>) – Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
  When passing tool names, tools are selected from the Agent's originally configured tools.
- **snapshot_callback** (<code>SnapshotCallback | None</code>) – Optional callback function that is invoked when a pipeline snapshot is created.
  The callback receives a `PipelineSnapshot` object and can return an optional string.
  If provided, the callback is used instead of the default file-saving behavior.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources
  to confirmation strategies. Useful in web/server environments to provide per-request
  objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
  can use for non-blocking user interaction.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.

**Raises:**

- <code>BreakpointException</code> – If an agent breakpoint is triggered.

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    *,
    generation_kwargs: dict[str, Any] | None = None,
    break_point: AgentBreakpoint | None = None,
    snapshot: AgentSnapshot | None = None,
    system_prompt: str | None = None,
    tools: ToolsType | list[str] | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None,
    **kwargs: Any
) -> dict[str, Any]
```

Asynchronously process messages and execute tools until the exit condition is met.

This is the asynchronous version of the `run` method. It follows the same logic but uses
asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
if available.

**Parameters:**

- **messages** (<code>list\[ChatMessage\]</code>) – List of Haystack ChatMessage objects to process.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – An asynchronous callback that will be invoked when a response is streamed from the
  LLM. The same callback can be configured to emit tool results when a tool is called.
- **generation_kwargs** (<code>dict\[str, Any\] | None</code>) – Additional keyword arguments for LLM. These parameters will
  override the parameters passed during component initialization.
- **break_point** (<code>AgentBreakpoint | None</code>) – An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
  for "tool_invoker".
- **snapshot** (<code>AgentSnapshot | None</code>) – A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
  the relevant information to restart the Agent execution from where it left off.
- **system_prompt** (<code>str | None</code>) – System prompt for the agent. If provided, it overrides the default system prompt.
- **tools** (<code>ToolsType | list\[str\] | None</code>) – Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
- **snapshot_callback** (<code>SnapshotCallback | None</code>) – Optional callback function that is invoked when a pipeline snapshot is created.
  The callback receives a `PipelineSnapshot` object and can return an optional string.
  If provided, the callback is used instead of the default file-saving behavior.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources
  to confirmation strategies. Useful in web/server environments to provide per-request
  objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
  can use for non-blocking user interaction.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.

**Raises:**

- <code>BreakpointException</code> – If an agent breakpoint is triggered.

## state/state

### State

State is a container for storing shared information during the execution of an Agent and its tools.

For instance, State can be used to store documents, context, and intermediate results.

Internally it wraps a `_data` dictionary defined by a `schema`. Each schema entry has:

```json
  "parameter_name": {
    "type": SomeType,  # expected type
    "handler": Optional[Callable[[Any, Any], Any]]  # merge/update function
  }
```

Handlers control how values are merged when using the `set()` method:

- For list types: defaults to `merge_lists` (concatenates lists)
- For other types: defaults to `replace_values` (overwrites existing value)

A `messages` field with type `list[ChatMessage]` is automatically added to the schema.

This makes it possible for the Agent to read from and write to the same context.

### Usage example

```python
from haystack.components.agents.state import State

my_state = State(
    schema={"gh_repo_name": {"type": str}, "user_name": {"type": str}},
    data={"gh_repo_name": "my_repo", "user_name": "my_user_name"}
)
```

#### __init__

```python
__init__(schema: dict[str, Any], data: dict[str, Any] | None = None)
```

Initialize a State object with a schema and optional data.

**Parameters:**

- **schema** (<code>dict\[str, Any\]</code>) – Dictionary mapping parameter names to their type and handler configs.
  Type must be a valid Python type, and handler must be a callable function or None.
  If handler is None, the default handler for the type will be used. The default handlers are:
  - For list types: `haystack.agents.state.state_utils.merge_lists`
  - For all other types: `haystack.agents.state.state_utils.replace_values`
- **data** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of initial data to populate the state

#### get

```python
get(key: str, default: Any = None) -> Any
```

Retrieve a value from the state by key.

**Parameters:**

- **key** (<code>str</code>) – Key to look up in the state
- **default** (<code>Any</code>) – Value to return if key is not found

**Returns:**

- <code>Any</code> – Value associated with key or default if not found

#### set

```python
set(
    key: str,
    value: Any,
    handler_override: Callable[[Any, Any], Any] | None = None,
) -> None
```

Set or merge a value in the state according to schema rules.

Value is merged or overwritten according to these rules:

- if handler_override is given, use that
- else use the handler defined in the schema for 'key'

**Parameters:**

- **key** (<code>str</code>) – Key to store the value under
- **value** (<code>Any</code>) – Value to store or merge
- **handler_override** (<code>Callable\\[[Any, Any\], Any\] | None</code>) – Optional function to override the default merge behavior

#### data

```python
data
```

All current data of the state.

#### has

```python
has(key: str) -> bool
```

Check if a key exists in the state.

**Parameters:**

- **key** (<code>str</code>) – Key to check for existence

**Returns:**

- <code>bool</code> – True if key exists in state, False otherwise

#### to_dict

```python
to_dict()
```

Convert the State object to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any])
```

Convert a dictionary back to a State object.
