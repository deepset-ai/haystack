---
title: "Hooks"
id: hooks-api
description: "Hooks that run at points in the Agent's run loop and influence it by mutating State, including built-in tool result offloading and Human-in-the-Loop tool confirmation."
slug: "/hooks-api"
---


## from_function

### FunctionHook

Wraps a function (or a sync/async pair) into a serializable `Hook`.

Produced by the `@hook` decorator for the single-function case. To give a hook both an optimized sync and async
path, construct it directly with both `function` and `async_function` set.

#### __init__

```python
__init__(
    function: Callable[[State], None] | None = None,
    async_function: Callable[[State], Awaitable[None]] | None = None,
) -> None
```

Initialize the hook with a synchronous function, an async function, or both.

**Parameters:**

- **function** (<code>Callable\\[[State\], None\] | None</code>) – The synchronous function invoked by `run`. Must be a regular function — coroutine functions
  should be passed to `async_function` instead. Either `function` or `async_function` (or both) must be set.
- **async_function** (<code>Callable\\[[State\], Awaitable[None]\] | None</code>) – Optional coroutine function awaited by `run_async`. When only `async_function` is set,
  `run` raises a `RuntimeError`. When only `function` is set, `run_async` calls `function`.

**Raises:**

- <code>ValueError</code> – If neither is set, if `function` is a coroutine function, if `async_function` is not, or
  if a provided function does not declare a `State`-typed parameter.

#### run

```python
run(state: State) -> None
```

Run the synchronous function against the live `State`.

**Parameters:**

- **state** (<code>State</code>) – The Agent's live `State`, mutated in place by the wrapped function.

**Raises:**

- <code>RuntimeError</code> – If the hook only has an `async_function`; use the Agent's async run methods instead.

#### run_async

```python
run_async(state: State) -> None
```

Await the async function if set, otherwise call the synchronous function.

**Parameters:**

- **state** (<code>State</code>) – The Agent's live `State`, mutated in place by the wrapped function.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the hook, storing each wrapped function as an importable reference.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the hook's type and the import paths of its sync/async functions.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FunctionHook
```

Deserialize the hook, resolving each function from its importable reference.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The serialized hook dictionary produced by `to_dict`.

**Returns:**

- <code>FunctionHook</code> – The reconstructed `FunctionHook`.

### hook

```python
hook(function: Callable[[State], None | Awaitable[None]]) -> FunctionHook
```

Wrap a function into a `Hook` the Agent can invoke during its run loop.

The decorated function receives the Agent's `State` and influences the run by mutating it in place. A coroutine
function is wrapped as the hook's async path; a regular function as its sync path. To give a single hook both
paths, construct a `FunctionHook` directly with both `function` and `async_function`.

### Usage example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.hooks import hook
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

@tool
def weather_tool(city: str) -> str:
    '''Get the current weather for a given city.'''
    return f"The weather in {city} is sunny."

@tool
def save(content: str) -> str:
    '''Save content to durable storage.'''
    return "Saved."

@hook
def require_save(state: State) -> None:
    if state.get("tool_call_counts", {}).get("save", 0) == 0:
        state.set("messages", [ChatMessage.from_system("You must call `save` before finishing.")])
        state.set("continue_run", True)

agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[weather_tool, save], hooks={"on_exit": [require_save]})
```

**Parameters:**

- **function** (<code>Callable\\[[State\], None | Awaitable[None]\]</code>) – A callable taking the Agent's `State` and returning `None` (sync or async).

**Returns:**

- <code>FunctionHook</code> – A `FunctionHook` wrapping the function.

## human_in_the_loop/dataclasses

### ConfirmationUIResult

Result of the confirmation UI interaction.

**Parameters:**

- **action** (<code>str</code>) – The action taken by the user such as "confirm", "reject", or "modify".
  This action type is not enforced to allow for custom actions to be implemented.
- **feedback** (<code>str | None</code>) – Optional feedback message from the user. For example, if the user rejects the tool execution,
  they might provide a reason for the rejection.
- **new_tool_params** (<code>dict\[str, Any\] | None</code>) – Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
  they can provide a new set of parameters here.

### ToolExecutionDecision

Decision made regarding tool execution.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **execute** (<code>bool</code>) – A boolean indicating whether to execute the tool with the provided parameters.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
  specific tool invocation.
- **feedback** (<code>str | None</code>) – Optional feedback message.
  For example, if the tool execution is rejected, this can contain the reason. Or if the tool parameters were
  modified, this can contain the modification details.
- **final_tool_params** (<code>dict\[str, Any\] | None</code>) – Optional final parameters for the tool if execution is confirmed or modified.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Convert the ToolExecutionDecision to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the tool execution decision details.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ToolExecutionDecision
```

Populate the ToolExecutionDecision from a dictionary representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary containing the tool execution decision details.

**Returns:**

- <code>ToolExecutionDecision</code> – An instance of ToolExecutionDecision.

## human_in_the_loop/hooks

### ConfirmationHook

A `before_tool` Agent hook that applies Human-in-the-Loop confirmation strategies to pending tool calls.

Register it on an `Agent` to confirm, modify, or reject tool calls before they run:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import tool
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    NeverAskPolicy,
    RichConsoleUI,
    SimpleConsoleUI,
)

@tool
def delete_file(path: str) -> str:
    '''Delete the file at the given path.'''
    return f"Deleted {path}."

hook = ConfirmationHook(
    confirmation_strategies={
        "delete_file": BlockingConfirmationStrategy(
            confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
        )
    }
)
agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[delete_file], hooks={"before_tool": [hook]})
```

A key may be a single tool name, a tuple of tool names sharing one strategy, or the wildcard `"*"` which applies
to any tool without a more specific entry. More specific keys win, so you can set a default for all tools and
override individual ones:

```python
hook = ConfirmationHook(
    confirmation_strategies={
        "delete_file": BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI()
        ),
        "*": BlockingConfirmationStrategy(
            confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
        ),
    }
)
```

Request-scoped resources for the strategies (e.g. a WebSocket or queue) are passed per run via the Agent's
`hook_context` argument (`agent.run(messages=[...], hook_context={...})`) and read by the hook with
`state.data.get("hook_context")`.

This hook only makes sense at the `before_tool` hook point, where the pending tool calls exist (between the model
requesting tools and those tools running); the Agent enforces this and raises if it is registered elsewhere. Use a
single ConfirmationHook with one entry per tool (or per tuple of tools) in `confirmation_strategies` rather than
registering several hooks.

#### __init__

```python
__init__(
    confirmation_strategies: dict[str | tuple[str, ...], ConfirmationStrategy],
) -> None
```

Initialize the hook with its per-tool confirmation strategies.

**Parameters:**

- **confirmation_strategies** (<code>dict\[str | tuple\[str, ...\], ConfirmationStrategy\]</code>) – Mapping of tool name (or a tuple of tool names) to its `ConfirmationStrategy`.
  The wildcard key `"*"` applies to any tool without a more specific entry.

#### run

```python
run(state: State) -> None
```

Confirm the pending tool calls, rewriting the `messages` in `state` to reflect modifications and rejections.

**Parameters:**

- **state** (<code>State</code>) – The Agent's live `State`. Reads the available tools (`state.data.get("tools")`) and the per-run
  context (`state.data.get("hook_context")`), and the pending tool calls from the last message; writes the
  updated conversation back to `messages`. Reads go through `state.data` rather than `state.get`, which
  deep-copies and would break non-copyable resources (e.g. a WebSocket or client) in `hook_context`.

#### run_async

```python
run_async(state: State) -> None
```

Async version of `run`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the hook, including its confirmation strategies (tuple keys become JSON-array strings).

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ConfirmationHook
```

Deserialize the hook, reconstructing its confirmation strategies.

## human_in_the_loop/policies

### AlwaysAskPolicy

Bases: <code>ConfirmationPolicy</code>

Always ask for confirmation.

#### should_ask

```python
should_ask(
    tool_name: str, tool_description: str, tool_params: dict[str, Any]
) -> bool
```

Always ask for confirmation before executing the tool.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – Always returns True, indicating confirmation is needed.

### NeverAskPolicy

Bases: <code>ConfirmationPolicy</code>

Never ask for confirmation.

#### should_ask

```python
should_ask(
    tool_name: str, tool_description: str, tool_params: dict[str, Any]
) -> bool
```

Never ask for confirmation, always proceed with tool execution.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – Always returns False, indicating no confirmation is needed.

### AskOncePolicy

Bases: <code>ConfirmationPolicy</code>

Ask only once per tool with specific parameters.

#### __init__

```python
__init__() -> None
```

Creates an instance of AskOncePolicy.

#### should_ask

```python
should_ask(
    tool_name: str, tool_description: str, tool_params: dict[str, Any]
) -> bool
```

Ask for confirmation only once per tool with specific parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – True if confirmation is needed, False if already asked with the same parameters.

#### update_after_confirmation

```python
update_after_confirmation(
    tool_name: str,
    tool_description: str,
    tool_params: dict[str, Any],
    confirmation_result: ConfirmationUIResult,
) -> None
```

Store the tool and parameters if the action was "confirm" to avoid asking again.

This method updates the internal state to remember that the user has already confirmed the execution of the
tool with the given parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that was executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters that were passed to the tool.
- **confirmation_result** (<code>ConfirmationUIResult</code>) – The result from the confirmation UI.

## human_in_the_loop/strategies

### BlockingConfirmationStrategy

Confirmation strategy that blocks execution to gather user feedback.

#### __init__

```python
__init__(
    *,
    confirmation_policy: ConfirmationPolicy,
    confirmation_ui: ConfirmationUI,
    reject_template: str = REJECTION_FEEDBACK_TEMPLATE,
    modify_template: str = MODIFICATION_FEEDBACK_TEMPLATE,
    user_feedback_template: str = USER_FEEDBACK_TEMPLATE
) -> None
```

Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

**Parameters:**

- **confirmation_policy** (<code>ConfirmationPolicy</code>) – The confirmation policy to determine when to ask for user confirmation.
- **confirmation_ui** (<code>ConfirmationUI</code>) – The user interface to interact with the user for confirmation.
- **reject_template** (<code>str</code>) – Template for rejection feedback messages. It should include a `{tool_name}` placeholder.
- **modify_template** (<code>str</code>) – Template for modification feedback messages. It should include `{tool_name}` and `{final_tool_params}`
  placeholders.
- **user_feedback_template** (<code>str</code>) – Template for user feedback messages. It should include a `{feedback}` placeholder.

#### run

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

Run the human-in-the-loop strategy for a given tool and its parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
  specific tool invocation.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources. Useful in web/server environments
  to provide per-request objects (e.g., WebSocket connections, async queues, Redis pub/sub clients)
  that strategies can use for non-blocking user interaction.

**Returns:**

- <code>ToolExecutionDecision</code> – A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
  feedback message if rejected.

#### run_async

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

Async version of run. Calls the sync run() method by default.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.
- **tool_call_id** (<code>str | None</code>) – Optional unique identifier for the tool call.
- **confirmation_strategy_context** (<code>dict\[str, Any\] | None</code>) – Optional dictionary for passing request-scoped resources.

**Returns:**

- <code>ToolExecutionDecision</code> – A ToolExecutionDecision indicating whether to execute the tool with the given parameters.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the BlockingConfirmationStrategy to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> BlockingConfirmationStrategy
```

Deserializes the BlockingConfirmationStrategy from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>BlockingConfirmationStrategy</code> – Deserialized BlockingConfirmationStrategy.

## human_in_the_loop/user_interfaces

### RichConsoleUI

Bases: <code>ConfirmationUI</code>

Rich console interface for user interaction.

#### __init__

```python
__init__(console: Console | None = None) -> None
```

Creates an instance of RichConsoleUI.

#### get_user_confirmation

```python
get_user_confirmation(
    tool_name: str, tool_description: str, tool_params: dict[str, Any]
) -> ConfirmationUIResult
```

Get user confirmation for tool execution via rich console prompts.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>ConfirmationUIResult</code> – ConfirmationUIResult based on user input.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the RichConsoleConfirmationUI to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### SimpleConsoleUI

Bases: <code>ConfirmationUI</code>

Simple console interface using standard input/output.

#### get_user_confirmation

```python
get_user_confirmation(
    tool_name: str, tool_description: str, tool_params: dict[str, Any]
) -> ConfirmationUIResult
```

Get user confirmation for tool execution via simple console prompts.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

## protocol

### Hook

Bases: <code>Protocol</code>

A callable the Agent invokes at a point in its run loop, receiving the live `State`.

A hook influences the run only by mutating `State` in place. At least `messages` (the conversation),
`step_count`, `token_usage` and `tool_call_counts` are available; any additional keys defined in the Agent's
`state_schema` are available too. The same hook object can be registered under multiple hook points.

Implement this protocol directly for stateful hooks (e.g. one wrapping a component), or use the `@hook` decorator to
wrap a plain `(State) -> None` function.

A hook may additionally define `async def run_async(self, state: State) -> None` for true async behavior; when
absent, the Agent calls `run` during async runs. It is left off this protocol on purpose so sync-only hooks
don't have to implement it.

A hook may also implement the optional lifecycle methods `warm_up` / `warm_up_async` and `close` / `close_async`.
The Agent calls them from its own `warm_up` / `warm_up_async` and `close` / `close_async`, so a hook can defer
opening clients or reading credentials until warm-up and release them on close.

#### run

```python
run(state: State) -> None
```

Run the hook against the live `State`, mutating it in place.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the hook to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> Hook
```

Deserialize the hook from a dictionary.

## tool_result_offloading/hooks

### ToolResultOffloadHook

Offload tool results to a `ToolResultStore`, replacing them in the conversation with a compact pointer.

This `after_tool` Agent hook writes the full result to the store so the next LLM call sees a reference instead of
the full result. Register it on an `Agent` under the `after_tool` hook point. Which tools offload, and under what
condition, is controlled per tool by `offload_strategies`:

<!-- test-concept -->

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.hooks.tool_result_offloading import (
    AlwaysOffload,
    FileSystemToolResultStore,
    NeverOffload,
    OffloadOverChars,
    ToolResultOffloadHook,
)

hook = ToolResultOffloadHook(
    store=FileSystemToolResultStore(root="tool_results"),
    offload_strategies={
        "web_search": AlwaysOffload(),          # force offload
        "get_time": NeverOffload(),             # opt out
        ("read_file", "list_dir"): OffloadOverChars(4000),  # tuple key: shared policy
        "*": OffloadOverChars(8000),            # wildcard default for any unlisted tool
    },
)
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
    tools=[web_search, get_time, read_file, list_dir],
    hooks={"after_tool": [hook]},
)
```

A key may be a single tool name, a tuple of tool names sharing one policy, or the wildcard `"*"` which applies to
any tool without a more specific entry. More specific keys win. A tool with no matching key (and no `"*"`) is not
offloaded.

Only successful, text tool output is offloaded. Error results (including `before_tool` human-in-the-loop
rejections) are always left in context. Non-text results (image or file content) are also left in context, and a
warning is logged when such a result has a matching offload policy; supporting only text is a deliberate choice
for now. Each result is offloaded at most once, even though the hook runs on every tool step.

The hook keeps no mutable state, so a single instance can be shared across concurrent runs. The constructor
`store`, however, is shared by every run that does not override it — fine for single-user or local use, but in a
multi-user server give each run its own isolated store (a per-session directory or sandbox) via `hook_context`
under the key `RESULT_STORE_CONTEXT_KEY`
(`agent.run(messages=[...], hook_context={RESULT_STORE_CONTEXT_KEY: per_request_store})`); it overrides the
constructor store for that run. Isolating the store per run keeps concurrent users from colliding on store keys or
reading each other's offloaded results — important especially when a bash/read tool is scoped to the store.

#### __init__

```python
__init__(
    store: ToolResultStore,
    offload_strategies: dict[str | tuple[str, ...], OffloadPolicy],
    *,
    preview_chars: int = 200
) -> None
```

Initialize the hook with a store and per-tool offload strategies.

**Parameters:**

- **store** (<code>ToolResultStore</code>) – Where offloaded results are written. Can be overridden per run via `hook_context`.
- **offload_strategies** (<code>dict\[str | tuple\[str, ...\], OffloadPolicy\]</code>) – Mapping of tool name (or a tuple of tool names, or the wildcard `"*"`) to the
  `OffloadPolicy` that decides whether that tool's results are offloaded.
- **preview_chars** (<code>int</code>) – Number of leading characters of the original result to include in the pointer left in
  the conversation, so the model knows roughly what was offloaded.

#### run

```python
run(state: State) -> None
```

Offload the freshly produced tool results in `state.data["messages"]` according to `offload_strategies`.

Considers only the trailing block of tool-result messages (the current step's results); earlier history is
left untouched. Offloads each of those messages its policy opts in for, and writes the rewritten conversation
back to `messages` only if at least one message changed.

Results are written to the store this run resolves to: a per-run store passed in `state`'s `hook_context`
under `RESULT_STORE_CONTEXT_KEY` if present, otherwise the store the hook was constructed with. Supply the
per-run store when calling the Agent, e.g.
`agent.run(messages=[...], hook_context={RESULT_STORE_CONTEXT_KEY: per_request_store})`. In a multi-user
server, pass an isolated store per run this way so concurrent users write to separate locations and never
read each other's results.

The hook keeps no mutable state, so a single instance is safe to share across concurrent runs; isolation
comes entirely from giving each run its own store via `hook_context`.

**Parameters:**

- **state** (<code>State</code>) – The Agent's live `State`. Reads the per-run store from `hook_context` and rewrites the offloaded
  tool-result messages back into `messages`.

**Returns:**

- <code>None</code> – None. The hook mutates `state` in place.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the hook, including its store and per-tool offload strategies.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the hook.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ToolResultOffloadHook
```

Deserialize the hook, reconstructing its store and offload strategies.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary representation produced by `to_dict`.

**Returns:**

- <code>ToolResultOffloadHook</code> – The deserialized `ToolResultOffloadHook`.

## tool_result_offloading/policies

### AlwaysOffload

Bases: <code>OffloadPolicy</code>

Offload every result of the tool it is assigned to.

#### should_offload

```python
should_offload(tool_name: str, result: str, state: State) -> bool
```

Decide whether to offload the given tool result.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that produced the result (unused; this policy always offloads).
- **result** (<code>str</code>) – The tool result string (unused; this policy always offloads).
- **state** (<code>State</code>) – The Agent's live `State` (unused; this policy always offloads).

**Returns:**

- <code>bool</code> – Always True.

### NeverOffload

Bases: <code>OffloadPolicy</code>

Never offload; keep the tool's full result in context. Use to opt a tool out of a wildcard default.

#### should_offload

```python
should_offload(tool_name: str, result: str, state: State) -> bool
```

Decide whether to offload the given tool result.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that produced the result (unused; this policy never offloads).
- **result** (<code>str</code>) – The tool result string (unused; this policy never offloads).
- **state** (<code>State</code>) – The Agent's live `State` (unused; this policy never offloads).

**Returns:**

- <code>bool</code> – Always False.

### OffloadOverChars

Bases: <code>OffloadPolicy</code>

Offload a result only when its string length exceeds `threshold` characters.

#### __init__

```python
__init__(threshold: int) -> None
```

Initialize the policy with its character threshold.

**Parameters:**

- **threshold** (<code>int</code>) – Offload the result when its length in characters is strictly greater than this value.

#### should_offload

```python
should_offload(tool_name: str, result: str, state: State) -> bool
```

Decide whether to offload the given tool result based on its length.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that produced the result (unused; only length is considered).
- **result** (<code>str</code>) – The tool result string whose length is compared against the threshold.
- **state** (<code>State</code>) – The Agent's live `State` (unused; only length is considered).

**Returns:**

- <code>bool</code> – True when `result` is longer than `threshold` characters, otherwise False.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the policy, including its threshold.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the policy.

## tool_result_offloading/stores

### FileSystemToolResultStore

Bases: <code>ToolResultStore</code>

A `ToolResultStore` that writes offloaded tool results to files under a root directory on the local file system.

```python
from haystack.hooks.tool_result_offloading import FileSystemToolResultStore

store = FileSystemToolResultStore(root="tool_results")
reference = store.write(key="search_1.txt", content="...")
store.read(reference)
```

#### __init__

```python
__init__(root: str | Path) -> None
```

Initialize the store with the root directory results are written under.

**Parameters:**

- **root** (<code>str | Path</code>) – Directory under which result files are written. Created on first write if it does not exist.

#### write

```python
write(*, key: str, content: str) -> str
```

Write `content` to `<root>/<key>`, creating parent directories, and return the file path.

The resolved target must stay within the root directory: a `key` that escapes it (e.g. containing `../` or an
absolute path) is rejected, so a tool-provided key cannot write outside the store.

**Parameters:**

- **key** (<code>str</code>) – Relative file name for the result within the store root.
- **content** (<code>str</code>) – The tool result to persist.

**Returns:**

- <code>str</code> – The absolute path the content was written to, as a string, for use with `read`.

**Raises:**

- <code>ValueError</code> – If `key` resolves to a location outside the store root.

#### read

```python
read(reference: str) -> str
```

Read back the content previously written to `reference`.

**Parameters:**

- **reference** (<code>str</code>) – A path returned by `write`.

**Returns:**

- <code>str</code> – The stored content.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the store, storing its root directory as a string.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary representation of the store.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> FileSystemToolResultStore
```

Deserialize the store from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary representation produced by `to_dict`.

**Returns:**

- <code>FileSystemToolResultStore</code> – The deserialized `FileSystemToolResultStore`.

## tool_result_offloading/types/protocol

### ToolResultStore

Bases: <code>Protocol</code>

A place a `ToolResultOffloadHook` writes offloaded tool results to, and reads them back from.

Implementations decide where and how the content lives (local disk, an isolated sandbox filesystem, object
storage, ...). `write` returns an opaque reference string that the Agent puts in the conversation in place of the
full result; `read` resolves that reference back to the original content.

Implement both `to_dict` and `from_dict` to make a custom store serializable; the default implementations below
cover stores whose constructor takes no arguments.

#### write

```python
write(*, key: str, content: str) -> str
```

Persist `content` under `key` and return an opaque reference to it.

**Parameters:**

- **key** (<code>str</code>) – A stable, per-result identifier the hook derives from the tool call (e.g. a file name).
- **content** (<code>str</code>) – The tool result to persist.

**Returns:**

- <code>str</code> – A reference string (e.g. a path or URI) that `read` can later resolve.

#### read

```python
read(reference: str) -> str
```

Return the content previously stored under `reference`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the store to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ToolResultStore
```

Deserialize the store from a dictionary.

### OffloadPolicy

Bases: <code>Protocol</code>

Decides, per tool result, whether the `ToolResultOffloadHook` offloads it to the store or leaves it in context.

A `ToolResultOffloadHook` maps tool names to policies, so different tools can offload under different conditions
(always, never, or a custom rule such as a size threshold).

Implement both `to_dict` and `from_dict` to make a custom policy serializable; the default implementations below
cover policies whose constructor takes no arguments.

#### should_offload

```python
should_offload(tool_name: str, result: str, state: State) -> bool
```

Return whether the given tool result should be offloaded.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that produced the result.
- **result** (<code>str</code>) – The tool result as a string (the content that would otherwise stay in the conversation).
- **state** (<code>State</code>) – The Agent's live `State`, for policies that decide based on run context.

**Returns:**

- <code>bool</code> – True to offload the result to the store, False to leave it in context.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OffloadPolicy
```

Deserialize the policy from a dictionary.
