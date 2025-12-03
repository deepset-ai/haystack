---
title: "Agents"
id: experimental-agents-api
description: "Tool-using agents with provider-agnostic chat model support."
slug: "/experimental-agents-api"
---

<a id="haystack_experimental.components.agents.agent"></a>

## Module haystack\_experimental.components.agents.agent

<a id="haystack_experimental.components.agents.agent.Agent"></a>

### Agent

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

<a id="haystack_experimental.components.agents.agent.Agent.__init__"></a>

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
             confirmation_strategies: Optional[dict[
                 str, ConfirmationStrategy]] = None,
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

<a id="haystack_experimental.components.agents.agent.Agent.run"></a>

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

<a id="haystack_experimental.components.agents.agent.Agent.run_async"></a>

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

<a id="haystack_experimental.components.agents.agent.Agent.to_dict"></a>

#### Agent.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the component to a dictionary.

**Returns**:

Dictionary with serialized data

<a id="haystack_experimental.components.agents.agent.Agent.from_dict"></a>

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

<a id="haystack_experimental.components.agents.human_in_the_loop.breakpoint"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.breakpoint

<a id="haystack_experimental.components.agents.human_in_the_loop.breakpoint.get_tool_calls_and_descriptions_from_snapshot"></a>

#### get\_tool\_calls\_and\_descriptions\_from\_snapshot

```python
def get_tool_calls_and_descriptions_from_snapshot(
        agent_snapshot: AgentSnapshot,
        breakpoint_tool_only: bool = True
) -> tuple[list[dict], dict[str, str]]
```

Extract tool calls and tool descriptions from an AgentSnapshot.

By default, only the tool call that caused the breakpoint is processed and its arguments are reconstructed.
This is useful for scenarios where you want to present the relevant tool call and its description
to a human for confirmation before execution.

**Arguments**:

- `agent_snapshot`: The AgentSnapshot from which to extract tool calls and descriptions.
- `breakpoint_tool_only`: If True, only the tool call that caused the breakpoint is returned. If False, all tool
calls are returned.

**Returns**:

A tuple containing a list of tool call dictionaries and a dictionary of tool descriptions

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.dataclasses

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses.ConfirmationUIResult"></a>

### ConfirmationUIResult

Result of the confirmation UI interaction.

**Arguments**:

- `action`: The action taken by the user such as "confirm", "reject", or "modify".
This action type is not enforced to allow for custom actions to be implemented.
- `feedback`: Optional feedback message from the user. For example, if the user rejects the tool execution,
they might provide a reason for the rejection.
- `new_tool_params`: Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
they can provide a new set of parameters here.

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses.ConfirmationUIResult.action"></a>

#### action

"confirm", "reject", "modify"

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses.ToolExecutionDecision"></a>

### ToolExecutionDecision

Decision made regarding tool execution.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `execute`: A boolean indicating whether to execute the tool with the provided parameters.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
specific tool invocation.
- `feedback`: Optional feedback message.
For example, if the tool execution is rejected, this can contain the reason. Or if the tool parameters were
modified, this can contain the modification details.
- `final_tool_params`: Optional final parameters for the tool if execution is confirmed or modified.

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses.ToolExecutionDecision.to_dict"></a>

#### ToolExecutionDecision.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the ToolExecutionDecision to a dictionary representation.

**Returns**:

A dictionary containing the tool execution decision details.

<a id="haystack_experimental.components.agents.human_in_the_loop.dataclasses.ToolExecutionDecision.from_dict"></a>

#### ToolExecutionDecision.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ToolExecutionDecision"
```

Populate the ToolExecutionDecision from a dictionary representation.

**Arguments**:

- `data`: A dictionary containing the tool execution decision details.

**Returns**:

An instance of ToolExecutionDecision.

<a id="haystack_experimental.components.agents.human_in_the_loop.errors"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.errors

<a id="haystack_experimental.components.agents.human_in_the_loop.errors.HITLBreakpointException"></a>

### HITLBreakpointException

Exception raised when a tool execution is paused by a ConfirmationStrategy (e.g. BreakpointConfirmationStrategy).

<a id="haystack_experimental.components.agents.human_in_the_loop.errors.HITLBreakpointException.__init__"></a>

#### HITLBreakpointException.\_\_init\_\_

```python
def __init__(message: str,
             tool_name: str,
             snapshot_file_path: str,
             tool_call_id: Optional[str] = None) -> None
```

Initialize the HITLBreakpointException.

**Arguments**:

- `message`: The exception message.
- `tool_name`: The name of the tool whose execution is paused.
- `snapshot_file_path`: The file path to the saved pipeline snapshot.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate
the decision with a specific tool invocation.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.policies

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.AlwaysAskPolicy"></a>

### AlwaysAskPolicy

Always ask for confirmation.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.AlwaysAskPolicy.should_ask"></a>

#### AlwaysAskPolicy.should\_ask

```python
def should_ask(tool_name: str, tool_description: str,
               tool_params: dict[str, Any]) -> bool
```

Always ask for confirmation before executing the tool.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.

**Returns**:

Always returns True, indicating confirmation is needed.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.NeverAskPolicy"></a>

### NeverAskPolicy

Never ask for confirmation.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.NeverAskPolicy.should_ask"></a>

#### NeverAskPolicy.should\_ask

```python
def should_ask(tool_name: str, tool_description: str,
               tool_params: dict[str, Any]) -> bool
```

Never ask for confirmation, always proceed with tool execution.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.

**Returns**:

Always returns False, indicating no confirmation is needed.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.AskOncePolicy"></a>

### AskOncePolicy

Ask only once per tool with specific parameters.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.AskOncePolicy.should_ask"></a>

#### AskOncePolicy.should\_ask

```python
def should_ask(tool_name: str, tool_description: str,
               tool_params: dict[str, Any]) -> bool
```

Ask for confirmation only once per tool with specific parameters.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.

**Returns**:

True if confirmation is needed, False if already asked with the same parameters.

<a id="haystack_experimental.components.agents.human_in_the_loop.policies.AskOncePolicy.update_after_confirmation"></a>

#### AskOncePolicy.update\_after\_confirmation

```python
def update_after_confirmation(
        tool_name: str, tool_description: str, tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult) -> None
```

Store the tool and parameters if the action was "confirm" to avoid asking again.

This method updates the internal state to remember that the user has already confirmed the execution of the
tool with the given parameters.

**Arguments**:

- `tool_name`: The name of the tool that was executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters that were passed to the tool.
- `confirmation_result`: The result from the confirmation UI.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.strategies

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy"></a>

### BlockingConfirmationStrategy

Confirmation strategy that blocks execution to gather user feedback.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy.__init__"></a>

#### BlockingConfirmationStrategy.\_\_init\_\_

```python
def __init__(confirmation_policy: ConfirmationPolicy,
             confirmation_ui: ConfirmationUI) -> None
```

Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

**Arguments**:

- `confirmation_policy`: The confirmation policy to determine when to ask for user confirmation.
- `confirmation_ui`: The user interface to interact with the user for confirmation.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy.run"></a>

#### BlockingConfirmationStrategy.run

```python
def run(tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: Optional[str] = None) -> ToolExecutionDecision
```

Run the human-in-the-loop strategy for a given tool and its parameters.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
specific tool invocation.

**Returns**:

A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
feedback message if rejected.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy.to_dict"></a>

#### BlockingConfirmationStrategy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the BlockingConfirmationStrategy to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy.from_dict"></a>

#### BlockingConfirmationStrategy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "BlockingConfirmationStrategy"
```

Deserializes the BlockingConfirmationStrategy from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized BlockingConfirmationStrategy.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy"></a>

### BreakpointConfirmationStrategy

Confirmation strategy that raises a tool breakpoint exception to pause execution and gather user feedback.

This strategy is designed for scenarios where immediate user interaction is not possible.
When a tool execution requires confirmation, it raises an `HITLBreakpointException`, which is caught by the Agent.
The Agent then serialize its current state, including the tool call details. This information can then be used to
notify a user to review and confirm the tool execution.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy.__init__"></a>

#### BreakpointConfirmationStrategy.\_\_init\_\_

```python
def __init__(snapshot_file_path: str) -> None
```

Initialize the BreakpointConfirmationStrategy.

**Arguments**:

- `snapshot_file_path`: The path to the directory that the snapshot should be saved.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy.run"></a>

#### BreakpointConfirmationStrategy.run

```python
def run(tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: Optional[str] = None) -> ToolExecutionDecision
```

Run the breakpoint confirmation strategy for a given tool and its parameters.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
specific tool invocation.

**Raises**:

- `HITLBreakpointException`: Always raises an `HITLBreakpointException` exception to signal that user confirmation is required.

**Returns**:

This method does not return; it always raises an exception.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy.to_dict"></a>

#### BreakpointConfirmationStrategy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the BreakpointConfirmationStrategy to a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy.from_dict"></a>

#### BreakpointConfirmationStrategy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "BreakpointConfirmationStrategy"
```

Deserializes the BreakpointConfirmationStrategy from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized BreakpointConfirmationStrategy.

<a id="haystack_experimental.components.agents.human_in_the_loop.types"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.types

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationUI"></a>

### ConfirmationUI

Base class for confirmation UIs.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationUI.get_user_confirmation"></a>

#### ConfirmationUI.get\_user\_confirmation

```python
def get_user_confirmation(tool_name: str, tool_description: str,
                          tool_params: dict[str, Any]) -> ConfirmationUIResult
```

Get user confirmation for tool execution.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationUI.to_dict"></a>

#### ConfirmationUI.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the UI to a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationUI.from_dict"></a>

#### ConfirmationUI.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI"
```

Deserialize the ConfirmationUI from a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationPolicy"></a>

### ConfirmationPolicy

Base class for confirmation policies.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationPolicy.should_ask"></a>

#### ConfirmationPolicy.should\_ask

```python
def should_ask(tool_name: str, tool_description: str,
               tool_params: dict[str, Any]) -> bool
```

Determine whether to ask for confirmation.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationPolicy.update_after_confirmation"></a>

#### ConfirmationPolicy.update\_after\_confirmation

```python
def update_after_confirmation(
        tool_name: str, tool_description: str, tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult) -> None
```

Update the policy based on the confirmation UI result.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationPolicy.to_dict"></a>

#### ConfirmationPolicy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationPolicy.from_dict"></a>

#### ConfirmationPolicy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy"
```

Deserialize the policy from a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationStrategy"></a>

### ConfirmationStrategy

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationStrategy.run"></a>

#### ConfirmationStrategy.run

```python
def run(tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: Optional[str] = None) -> ToolExecutionDecision
```

Run the confirmation strategy for a given tool and its parameters.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate
the decision with a specific tool invocation.

**Returns**:

The result of the confirmation strategy (e.g., tool output, rejection message, etc.).

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationStrategy.to_dict"></a>

#### ConfirmationStrategy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the strategy to a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.types.ConfirmationStrategy.from_dict"></a>

#### ConfirmationStrategy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationStrategy"
```

Deserialize the strategy from a dictionary.

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces"></a>

## Module haystack\_experimental.components.agents.human\_in\_the\_loop.user\_interfaces

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces.RichConsoleUI"></a>

### RichConsoleUI

Rich console interface for user interaction.

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces.RichConsoleUI.get_user_confirmation"></a>

#### RichConsoleUI.get\_user\_confirmation

```python
def get_user_confirmation(tool_name: str, tool_description: str,
                          tool_params: dict[str, Any]) -> ConfirmationUIResult
```

Get user confirmation for tool execution via rich console prompts.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.

**Returns**:

ConfirmationUIResult based on user input.

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces.RichConsoleUI.to_dict"></a>

#### RichConsoleUI.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the RichConsoleConfirmationUI to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI"></a>

### SimpleConsoleUI

Simple console interface using standard input/output.

<a id="haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI.get_user_confirmation"></a>

#### SimpleConsoleUI.get\_user\_confirmation

```python
def get_user_confirmation(tool_name: str, tool_description: str,
                          tool_params: dict[str, Any]) -> ConfirmationUIResult
```

Get user confirmation for tool execution via simple console prompts.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.

