---
title: "Agents"
id: agents-api
description: "Tool-using agents with provider-agnostic chat model support."
slug: "/agents-api"
---

<a id="agent"></a>

## Module agent

<a id="agent.Agent"></a>

### Agent

A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

The component processes messages and executes tools until an exit condition is met.
The exit condition can be triggered either by a direct text response or by invoking a specific designated tool.
Multiple exit conditions can be specified.

When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

### Usage example
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

# The agent will:
# 1. Search for tipping customs in France
# 2. Use calculator to compute tip based on findings
# 3. Return the final answer with context
print(result["messages"][-1].text)
```

<a id="agent.Agent.__init__"></a>

#### Agent.\_\_init\_\_

```python
def __init__(*,
             chat_generator: ChatGenerator,
             tools: Optional[ToolsType] = None,
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
- `tools`: A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
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
        generation_kwargs: Optional[dict[str, Any]] = None,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: Any) -> dict[str, Any]
```

Process messages and execute tools until an exit condition is met.

**Arguments**:

- `messages`: List of Haystack ChatMessage objects to process.
- `streaming_callback`: A callback that will be invoked when a response is streamed from the LLM.
The same callback can be configured to emit tool results when a tool is called.
- `generation_kwargs`: Additional keyword arguments for LLM. These parameters will
override the parameters passed during component initialization.
- `break_point`: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
for "tool_invoker".
- `snapshot`: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
the relevant information to restart the Agent execution from where it left off.
- `system_prompt`: System prompt for the agent. If provided, it overrides the default system prompt.
- `tools`: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
When passing tool names, tools are selected from the Agent's originally configured tools.
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
                    generation_kwargs: Optional[dict[str, Any]] = None,
                    break_point: Optional[AgentBreakpoint] = None,
                    snapshot: Optional[AgentSnapshot] = None,
                    system_prompt: Optional[str] = None,
                    tools: Optional[Union[ToolsType, list[str]]] = None,
                    **kwargs: Any) -> dict[str, Any]
```

Asynchronously process messages and execute tools until the exit condition is met.

This is the asynchronous version of the `run` method. It follows the same logic but uses
asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
if available.

**Arguments**:

- `messages`: List of Haystack ChatMessage objects to process.
- `streaming_callback`: An asynchronous callback that will be invoked when a response is streamed from the
LLM. The same callback can be configured to emit tool results when a tool is called.
- `generation_kwargs`: Additional keyword arguments for LLM. These parameters will
override the parameters passed during component initialization.
- `break_point`: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
for "tool_invoker".
- `snapshot`: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
the relevant information to restart the Agent execution from where it left off.
- `system_prompt`: System prompt for the agent. If provided, it overrides the default system prompt.
- `tools`: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
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

<a id="state/state"></a>

## Module state/state

<a id="state/state.State"></a>

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

<a id="state/state.State.__init__"></a>

#### State.\_\_init\_\_

```python
def __init__(schema: dict[str, Any], data: Optional[dict[str, Any]] = None)
```

Initialize a State object with a schema and optional data.

**Arguments**:

- `schema`: Dictionary mapping parameter names to their type and handler configs.
Type must be a valid Python type, and handler must be a callable function or None.
If handler is None, the default handler for the type will be used. The default handlers are:
    - For list types: `haystack.agents.state.state_utils.merge_lists`
    - For all other types: `haystack.agents.state.state_utils.replace_values`
- `data`: Optional dictionary of initial data to populate the state

<a id="state/state.State.get"></a>

#### State.get

```python
def get(key: str, default: Any = None) -> Any
```

Retrieve a value from the state by key.

**Arguments**:

- `key`: Key to look up in the state
- `default`: Value to return if key is not found

**Returns**:

Value associated with key or default if not found

<a id="state/state.State.set"></a>

#### State.set

```python
def set(key: str,
        value: Any,
        handler_override: Optional[Callable[[Any, Any], Any]] = None) -> None
```

Set or merge a value in the state according to schema rules.

Value is merged or overwritten according to these rules:
  - if handler_override is given, use that
  - else use the handler defined in the schema for 'key'

**Arguments**:

- `key`: Key to store the value under
- `value`: Value to store or merge
- `handler_override`: Optional function to override the default merge behavior

<a id="state/state.State.data"></a>

#### State.data

```python
@property
def data()
```

All current data of the state.

<a id="state/state.State.has"></a>

#### State.has

```python
def has(key: str) -> bool
```

Check if a key exists in the state.

**Arguments**:

- `key`: Key to check for existence

**Returns**:

True if key exists in state, False otherwise

<a id="state/state.State.to_dict"></a>

#### State.to\_dict

```python
def to_dict()
```

Convert the State object to a dictionary.

<a id="state/state.State.from_dict"></a>

#### State.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any])
```

Convert a dictionary back to a State object.

