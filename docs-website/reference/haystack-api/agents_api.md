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
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from typing import Annotated, Literal

# Tool functions - in practice, these would have real implementations
@tool
def search(query: Annotated[str, "The search query"]) -> str:
    '''Search for information on the web.'''
    # Placeholder: would call actual search API
    return "In France, a 15% service charge is typically included, but leaving 5-10% extra is appreciated."

@tool
def calculator(
    operation: Annotated[Literal["multiply", "percentage"], "The mathematical operation to perform"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"],
) -> float:
    '''Perform mathematical calculations.'''
    if operation == "multiply":
        return a * b
    elif operation == "percentage":
        return (a / 100) * b
    return 0

agent = Agent(
    system_prompt=(
        "You are a helpful assistant. Use the 'search' tool to find information "
        "about a user's question and the 'calculator' tool to perform math."
    ),
    chat_generator=OpenAIChatGenerator(),
    tools=[search, calculator],
    streaming_callback=print_streaming_chunk,
)

result = agent.run(
    messages=[ChatMessage.from_user("Calculate the appropriate tip for an €85 meal in France")]
)

# Access the final response from the Agent
# print(result["last_message"].text)
```

#### Using a `user_prompt` template with variables

You can define a reusable `user_prompt` with Jinja2 template variables so the Agent can be invoked
with different inputs without manually constructing `ChatMessage` objects each time.
This is especially useful when embedding the Agent in a pipeline.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import tool
from typing import Annotated


@tool
def translate(
    text: Annotated[str, "The text to translate"],
    target_language: Annotated[str, "The language to translate to"],
) -> str:
    """Translate text to a target language."""
    # Placeholder: would call an actual translation API
    return f"[Translated '{text}' to {target_language}]"

agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[translate],
    system_prompt="You are a helpful translation assistant.",
    user_prompt="""{% message role="user"%}
Translate the following document to {{ language }}: {{ document }}
{% endmessage %}""",
)

# The template variables 'language' and 'document' become inputs to the run method
result = agent.run(
    messages=[],
    language="French",
    document="The weather is lovely today and the sun is shining.",
)

print(result["last_message"].text)
```

#### Using hooks to influence the run loop

Hooks are callables that receive the live `State` and run at specific points in the Agent loop:

- `before_llm`: runs before each chat-generator call.
- `before_tool`: runs after the model requests tool calls, before any tools run. After these hooks run, the Agent
  re-reads the current last message from `state.data["messages"]`. If that message has tool calls, those calls are
  executed. If it has no tool calls, no tools run for that step, no tool-based exit condition is triggered, and the
  Agent loops back to the next LLM call unless `max_agent_steps` has been reached.
- `after_tool`: runs after tools execute, once their result messages are in `state.data["messages"]`, before the
  exit check and the next LLM call. Use it to rewrite the freshly produced tool-result messages (e.g. offload,
  redact, truncate, or summarize results). It does not run on the plain-text exit step, where no tools run.
- `on_exit`: runs when the Agent is about to stop on an exit condition. An `on_exit` hook can keep the Agent
  running by setting `state.set("continue_run", True)`.

Use the `@hook` decorator to build a hook from a function. This `on_exit` hook keeps the Agent running until a
required tool has been called.

```python
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks import hook
from haystack.tools import tool
from typing import Annotated


@tool
def save_result(content: Annotated[str, "The result to save"]) -> str:
    """Save the final result."""
    # Placeholder: would persist `content` to a database or the file system
    return "saved"


@hook
def require_save(state: State) -> None:
    if state.get("tool_call_counts", {}).get("save_result", 0) == 0:
        state.set("messages", [ChatMessage.from_system("Call `save_result` before finishing.")])
        state.set("continue_run", True)  # keep the Agent running instead of stopping


agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[save_result],
    hooks={"on_exit": [require_save]},
)
```

#### __init__

```python
__init__(
    *,
    chat_generator: ChatGenerator,
    tools: ToolsType | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    required_variables: list[str] | Literal["*"] | None = "*",
    exit_conditions: list[str] | None = None,
    state_schema: dict[str, Any] | None = None,
    max_agent_steps: int = 100,
    streaming_callback: StreamingCallbackT | None = None,
    raise_on_tool_invocation_failure: bool = False,
    tool_concurrency_limit: int = 4,
    tool_streaming_callback_passthrough: bool = False,
    hooks: dict[HookPoint, list[Hook]] | None = None
) -> None
```

Initialize the agent component.

**Parameters:**

- **chat_generator** (<code>ChatGenerator</code>) – An instance of the chat generator that your agent should use. It must support tools.
- **tools** (<code>ToolsType | None</code>) – A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
- **system_prompt** (<code>str | None</code>) – System prompt for the agent. Can be a plain string template or a Jinja2 message template.
  For details on the supported template syntax, refer to the
  [documentation](https://docs.haystack.deepset.ai/docs/chatpromptbuilder#string-templates).
- **user_prompt** (<code>str | None</code>) – User prompt for the agent. Can be a plain string template or a Jinja2 message template.
  If provided, this is appended to the messages provided at runtime.
  For details on the supported template syntax, refer to the
  [documentation](https://docs.haystack.deepset.ai/docs/chatpromptbuilder#string-templates).
- **required_variables** (<code>list\[str\] | Literal['\*'] | None</code>) – Lists the variables that must be provided as inputs to `user_prompt` or `system_prompt`.
  If a required variable is not provided at run time, an exception is raised.
  If set to `"*"`, all variables found in the prompts are required. Defaults to `"*"`.
  Set to `None` to make all variables optional; missing ones render as empty strings.
- **exit_conditions** (<code>list\[str\] | None</code>) – List of conditions that will cause the agent to return.
  Can include "text" if the agent should return when it generates a message without tool calls,
  or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
- **state_schema** (<code>dict\[str, Any\] | None</code>) – A dictionary defining the agent's runtime state. Each key maps to a type config
  with `"type"` (required) and an optional `"handler"` for merging values across tool calls.
  Tools can read from and write to state keys using `inputs_from_state` and `outputs_to_state`.
- **max_agent_steps** (<code>int</code>) – Maximum number of steps the agent will run before stopping. Defaults to 100.
  A step is one chat-generator call plus the execution of every tool call the model requested in
  that call (if any). If the agent reaches this number of steps it stops and returns the current state.
- **streaming_callback** (<code>StreamingCallbackT | None</code>) – A callback that will be invoked when a response is streamed from the LLM.
  The same callback can be configured to emit tool results when a tool is called.
- **raise_on_tool_invocation_failure** (<code>bool</code>) – Should the agent raise an exception when a tool invocation fails?
  If set to False, the exception will be turned into a chat message and passed to the LLM.
- **tool_concurrency_limit** (<code>int</code>) – Maximum number of tool calls to execute at the same time.
  Defaults to 4. Set to 1 to disable parallel tool execution.
- **tool_streaming_callback_passthrough** (<code>bool</code>) – If True, pass the streaming callback to tools that accept it.
- **hooks** (<code>dict\[HookPoint, list\[Hook\]\] | None</code>) – A dictionary mapping a hook point to a list of hooks the Agent runs at that point. Each hook
  receives the live `State` and influences the run by mutating it in place; hooks for a hook point run in
  list order. Valid hook points are:
- "before_run": Runs once per run, after the state is initialized and before the first chat-generator
  call. Use it to rewrite the initial messages or seed state (e.g. turn the user query into a task
  brief) without re-running on every step like "before_llm" does.
- "before_llm": Runs before each chat-generator call.
- "before_tool": Runs after the model requests tool calls, before any tools run. After these hooks run,
  the Agent re-reads the current last message from `state.data["messages"]`. If that message contains tool
  calls, those calls are executed. If it does not, no tools run for that step, no tool-based exit condition
  is triggered, and the Agent loops back to the next LLM call unless `max_agent_steps` has been reached.
- "after_tool": Runs after tools execute, once their result messages are in `state.data["messages"]`,
  before the exit check and the next LLM call. Use it to rewrite the freshly produced tool-result messages
  (e.g. offload, redact, truncate, or summarize results). It does not run on the plain-text exit step,
  where no tools run.
- "on_exit": Runs when the Agent is about to stop on an exit condition. An "on_exit" hook can keep the
  Agent running by setting the `continue_run` control flag (`state.set("continue_run", True)`), usually
  alongside a message telling the model what to do next. "on_exit" hooks run when the Agent stops on an
  exit condition, but not when it stops because `max_agent_steps` is reached.
- "after_run": Runs once per run, after the step loop has ended and before the Agent builds its return
  value — regardless of whether the run stopped on an exit condition or because `max_agent_steps` was
  reached (unlike "on_exit"). Mutations to the state (e.g. appending a final message) are reflected in
  the returned `messages` / `last_message` and `state_schema` outputs. Setting `continue_run` here has
  no effect.

**Raises:**

- <code>TypeError</code> – If the chat_generator does not support tools parameter in its run method.
- <code>ValueError</code> – If any `user_prompt` variable overlaps with the `state_schema` or `run` method parameters,
  if a hook is registered under an unknown hook point, or if a hook is registered under a hook point it does
  not support (via its `allowed_hook_points`).

#### warm_up

```python
warm_up() -> None
```

Warm up the tools, hooks, and the underlying chat generator.

#### warm_up_async

```python
warm_up_async() -> None
```

Warm up the tools, hooks, and the underlying chat generator on the serving event loop.

#### close

```python
close() -> None
```

Release the hooks' and the underlying chat generator's resources.

#### close_async

```python
close_async() -> None
```

Release the hooks' and the underlying chat generator's async resources.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Agent
```

Deserialize the agent from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>Agent</code> – Deserialized agent.

#### run

```python
run(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    *,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | list[str] | None = None,
    hook_context: dict[str, Any] | None = None,
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
- **tools** (<code>ToolsType | list\[str\] | None</code>) – Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
  When passing tool names, tools are selected from the Agent's originally configured tools.
- **hook_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of request-scoped resources made available to hooks via
  `state.data.get("hook_context")`. Useful in web/server environments to provide per-request objects
  (e.g., WebSocket connections, async queues, Redis pub/sub clients) that a hook can use, for
  example a ConfirmationHook driving non-blocking user interaction.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- "step_count": The number of steps the agent ran. A step is one chat-generator call plus the
  execution of every tool call the model requested in that call (if any). The counter is incremented
  after each step completes, including the final step that hits an exit condition or `max_agent_steps`.
- "token_usage": Aggregated token usage from every LLM call in the run, summed from each LLM message's
  `meta["usage"]`.
- "tool_call_counts": Mapping of tool name to the number of times that tool was invoked.
- "exit_reason": Why the Agent stopped, useful for routing the output downstream (e.g. with a
  `ConditionalRouter`). One of: `"text"` (the model returned a reply with no tool calls), the name of
  the tool that satisfied a tool exit condition (in which case `last_message` is that tool's result),
  or `"max_agent_steps"` (the Agent hit `max_agent_steps` before meeting an exit condition).
- Any additional keys defined in the `state_schema`.

#### run_async

```python
run_async(
    messages: list[ChatMessage],
    streaming_callback: StreamingCallbackT | None = None,
    *,
    generation_kwargs: dict[str, Any] | None = None,
    tools: ToolsType | list[str] | None = None,
    hook_context: dict[str, Any] | None = None,
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
- **tools** (<code>ToolsType | list\[str\] | None</code>) – Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
- **hook_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary of request-scoped resources made available to hooks via
  `state.data.get("hook_context")`. Useful in web/server environments to provide per-request objects
  (e.g., WebSocket connections, async queues, Redis pub/sub clients) that a hook can use, for
  example a ConfirmationHook driving non-blocking user interaction.
- **kwargs** (<code>Any</code>) – Additional data to pass to the State schema used by the Agent.
  The keys must match the schema defined in the Agent's `state_schema`.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the following keys:
- "messages": List of all messages exchanged during the agent's run.
- "last_message": The last message exchanged during the agent's run.
- "step_count": The number of steps the agent ran. A step is one chat-generator call plus the
  execution of every tool call the model requested in that call (if any). The counter is incremented
  after each step completes, including the final step that hits an exit condition or `max_agent_steps`.
- "token_usage": Aggregated token usage from every LLM call in the run, summed from each LLM message's
  `meta["usage"]`.
- "tool_call_counts": Mapping of tool name to the number of times that tool was invoked.
- "exit_reason": Why the Agent stopped, useful for routing the output downstream (e.g. with a
  `ConditionalRouter`). One of: `"text"` (the model returned a reply with no tool calls), the name of
  the tool that satisfied a tool exit condition (in which case `last_message` is that tool's result),
  or `"max_agent_steps"` (the Agent hit `max_agent_steps` before meeting an exit condition).
- Any additional keys defined in the `state_schema`.

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
__init__(schema: dict[str, Any], data: dict[str, Any] | None = None) -> None
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
data: dict[str, Any]
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
to_dict() -> dict[str, Any]
```

Convert the State object to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> State
```

Convert a dictionary back to a State object.
