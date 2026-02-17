---
title: "Agents"
id: experimental-agents-api
description: "Tool-using agents with provider-agnostic chat model support."
slug: "/experimental-agents-api"
---


## `haystack-experimental.haystack_experimental.components.agents.agent`

### `Agent`

Bases: <code>Agent</code>

A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

NOTE: This class extends Haystack's Agent component to add support for human-in-the-loop confirmation strategies.

The component processes messages and executes tools until an exit condition is met.
The exit condition can be triggered either by a direct text response or by invoking a specific designated tool.
Multiple exit conditions can be specified.

When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

### Usage example

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool

from haystack_experimental.components.agents import Agent
from haystack_experimental.components.agents.human_in_the_loop import (
    HumanInTheLoopStrategy,
    AlwaysAskPolicy,
    NeverAskPolicy,
    SimpleConsoleUI,
)

calculator_tool = Tool(name="calculator", description="A tool for performing mathematical calculations.", ...)
search_tool = Tool(name="search", description="A tool for searching the web.", ...)

agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[calculator_tool, search_tool],
    confirmation_strategies={
        calculator_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
        ),
        search_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
        ),
    },
)

# Run the agent
result = agent.run(
    messages=[ChatMessage.from_user("Find information about Haystack")]
)

assert "messages" in result  # Contains conversation history
```

#### `__init__`

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
    confirmation_strategies: dict[str, ConfirmationStrategy] | None = None,
    tool_invoker_kwargs: dict[str, Any] | None = None,
    chat_message_store: ChatMessageStore | None = None,
    memory_store: MemoryStore | None = None
) -> None
```

Initialize the agent component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator</code>) – An instance of the chat generator that your agent should use. It must support tools.
- **tools** (<code>ToolsType | None</code>) – List of Tool objects or a Toolset that the agent can use.
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
- **chat_message_store** (<code>ChatMessageStore | None</code>) – The ChatMessageStore that the agent can use to store
  and retrieve chat messages history.
- **memory_store** (<code>MemoryStore | None</code>) – The memory store that the agent can use to store and retrieve memories.

**Raises:**

- <code>TypeError</code> – If the chat_generator does not support tools parameter in its run method.
- <code>ValueError</code> – If the exit_conditions are not valid.

#### `run`

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
    confirmation_strategy_context: dict[str, Any] | None = None,
    chat_message_store_kwargs: dict[str, Any] | None = None,
    memory_store_kwargs: dict[str, Any] | None = None,
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
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources
  to confirmation strategies. Useful in web/server environments to provide per-request
  objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
  can use for non-blocking user interaction.
- **chat_message_store_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of keyword arguments to pass to the ChatMessageStore.
  For example, it can include the `chat_history_id` and `last_k` parameters for retrieving chat history.
- **memory_store_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of keyword arguments to pass to the MemoryStore.
  It can include:
- `user_id`: The user ID to search and add memories from.
- `run_id`: The run ID to search and add memories from.
- `agent_id`: The agent ID to search and add memories from.
- `search_criteria`: A dictionary of containing kwargs for the `search_memories` method.
  This can include:
  - `filters`: A dictionary of filters to search for memories.
  - `query`: The query to search for memories.
    Note: If you pass this, the user query passed to the agent will be
    ignored for memory retrieval.
  - `top_k`: The number of memories to return.
  - `include_memory_metadata`: Whether to include the memory metadata in the ChatMessage.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.

**Raises:**

- <code>RuntimeError</code> – If the Agent component wasn't warmed up before calling `run()`.
- <code>BreakpointException</code> – If an agent breakpoint is triggered.

#### `run_async`

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
    confirmation_strategy_context: dict[str, Any] | None = None,
    chat_message_store_kwargs: dict[str, Any] | None = None,
    memory_store_kwargs: dict[str, Any] | None = None,
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
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources
  to confirmation strategies. Useful in web/server environments to provide per-request
  objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
  can use for non-blocking user interaction.
- **chat_message_store_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of keyword arguments to pass to the ChatMessageStore.
  For example, it can include the `chat_history_id` and `last_k` parameters for retrieving chat history.
- **memory_store_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of keyword arguments to pass to the MemoryStore.
  It can include:
- `user_id`: The user ID to search and add memories from.
- `run_id`: The run ID to search and add memories from.
- `agent_id`: The agent ID to search and add memories from.
- `search_criteria`: A dictionary of containing kwargs for the `search_memories` method.
  This can include:
  - `filters`: A dictionary of filters to search for memories.
  - `query`: The query to search for memories.
    Note: If you pass this, the user query passed to the agent will be
    ignored for memory retrieval.
  - `top_k`: The number of memories to return.
  - `include_memory_metadata`: Whether to include the memory metadata in the ChatMessage.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- Any additional keys defined in the `state_schema`.

**Raises:**

- <code>RuntimeError</code> – If the Agent component wasn't warmed up before calling `run_async()`.
- <code>BreakpointException</code> – If an agent breakpoint is triggered.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> Agent
```

Deserialize the agent from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from

**Returns:**

- <code>Agent</code> – Deserialized agent

## `haystack-experimental.haystack_experimental.components.agents.human_in_the_loop.breakpoint`

### `get_tool_calls_and_descriptions_from_snapshot`

```python
get_tool_calls_and_descriptions_from_snapshot(
    agent_snapshot: AgentSnapshot, breakpoint_tool_only: bool = True
) -> tuple[list[dict], dict[str, str]]
```

Extract tool calls and tool descriptions from an AgentSnapshot.

By default, only the tool call that caused the breakpoint is processed and its arguments are reconstructed.
This is useful for scenarios where you want to present the relevant tool call and its description
to a human for confirmation before execution.

**Parameters:**

- **agent_snapshot** (<code>AgentSnapshot</code>) – The AgentSnapshot from which to extract tool calls and descriptions.
- **breakpoint_tool_only** (<code>bool</code>) – If True, only the tool call that caused the breakpoint is returned. If False, all tool
  calls are returned.

**Returns:**

- <code>tuple\[list\[dict\], dict\[str, str\]\]</code> – A tuple containing a list of tool call dictionaries and a dictionary of tool descriptions

## `haystack-experimental.haystack_experimental.components.agents.human_in_the_loop.errors`

### `HITLBreakpointException`

Bases: <code>Exception</code>

Exception raised when a tool execution is paused by a ConfirmationStrategy (e.g. BreakpointConfirmationStrategy).

#### `__init__`

```python
__init__(
    message: str,
    tool_name: str,
    snapshot_file_path: str,
    tool_call_id: str | None = None,
) -> None
```

Initialize the HITLBreakpointException.

**Parameters:**

- **message** (<code>str</code>) – The exception message.
- **tool_name** (<code>str</code>) – The name of the tool whose execution is paused.
- **snapshot_file_path** (<code>str</code>) – The file path to the saved pipeline snapshot.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call. This can be used to track and correlate
  the decision with a specific tool invocation.

## `haystack-experimental.haystack_experimental.components.agents.human_in_the_loop.strategies`

### `BreakpointConfirmationStrategy`

Confirmation strategy that raises a tool breakpoint exception to pause execution and gather user feedback.

This strategy is designed for scenarios where immediate user interaction is not possible.
When a tool execution requires confirmation, it raises an `HITLBreakpointException`, which is caught by the Agent.
The Agent then serialize its current state, including the tool call details. This information can then be used to
notify a user to review and confirm the tool execution.

#### `__init__`

```python
__init__(snapshot_file_path: str) -> None
```

Initialize the BreakpointConfirmationStrategy.

**Parameters:**

- **snapshot_file_path** (<code>str</code>) – The path to the directory that the snapshot should be saved.

#### `run`

```python
run(
    *,
    tool_name: str,
    tool_description: str,
    tool_params: dict[str, Any],
    tool_call_id: str | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None
) -> ToolExecutionDecision
```

Run the breakpoint confirmation strategy for a given tool and its parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
  specific tool invocation.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources. Not used by this strategy but included for
  interface compatibility.

**Returns:**

- <code>ToolExecutionDecision</code> – This method does not return; it always raises an exception.

**Raises:**

- <code>HITLBreakpointException</code> – Always raises an `HITLBreakpointException` exception to signal that user confirmation is required.

#### `run_async`

```python
run_async(
    *,
    tool_name: str,
    tool_description: str,
    tool_params: dict[str, Any],
    tool_call_id: str | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None
) -> ToolExecutionDecision
```

Async version of run. Calls the sync run() method.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources.

**Returns:**

- <code>ToolExecutionDecision</code> – This method does not return; it always raises an exception.

**Raises:**

- <code>HITLBreakpointException</code> – Always raises an `HITLBreakpointException` exception to signal that user confirmation is required.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the BreakpointConfirmationStrategy to a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> BreakpointConfirmationStrategy
```

Deserializes the BreakpointConfirmationStrategy from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>BreakpointConfirmationStrategy</code> – Deserialized BreakpointConfirmationStrategy.
