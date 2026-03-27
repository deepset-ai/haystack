---
title: "Human-in-the-Loop"
id: human-in-the-loop-api
description: "Abstractions for integrating human feedback and interaction into Agent workflows."
slug: "/human-in-the-loop-api"
---

<a id="dataclasses"></a>

## Module dataclasses

<a id="dataclasses.ConfirmationUIResult"></a>

### ConfirmationUIResult

Result of the confirmation UI interaction.

**Arguments**:

- `action`: The action taken by the user such as "confirm", "reject", or "modify".
This action type is not enforced to allow for custom actions to be implemented.
- `feedback`: Optional feedback message from the user. For example, if the user rejects the tool execution,
they might provide a reason for the rejection.
- `new_tool_params`: Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
they can provide a new set of parameters here.

<a id="dataclasses.ConfirmationUIResult.action"></a>

#### action

"confirm", "reject", "modify"

<a id="dataclasses.ToolExecutionDecision"></a>

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

<a id="dataclasses.ToolExecutionDecision.to_dict"></a>

#### ToolExecutionDecision.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Convert the ToolExecutionDecision to a dictionary representation.

**Returns**:

A dictionary containing the tool execution decision details.

<a id="dataclasses.ToolExecutionDecision.from_dict"></a>

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

<a id="policies"></a>

## Module policies

<a id="policies.AlwaysAskPolicy"></a>

### AlwaysAskPolicy

Always ask for confirmation.

<a id="policies.AlwaysAskPolicy.should_ask"></a>

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

<a id="policies.AlwaysAskPolicy.update_after_confirmation"></a>

#### AlwaysAskPolicy.update\_after\_confirmation

```python
def update_after_confirmation(
        tool_name: str, tool_description: str, tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult) -> None
```

Update the policy based on the confirmation UI result.

<a id="policies.AlwaysAskPolicy.to_dict"></a>

#### AlwaysAskPolicy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

<a id="policies.AlwaysAskPolicy.from_dict"></a>

#### AlwaysAskPolicy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy"
```

Deserialize the policy from a dictionary.

<a id="policies.NeverAskPolicy"></a>

### NeverAskPolicy

Never ask for confirmation.

<a id="policies.NeverAskPolicy.should_ask"></a>

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

<a id="policies.NeverAskPolicy.update_after_confirmation"></a>

#### NeverAskPolicy.update\_after\_confirmation

```python
def update_after_confirmation(
        tool_name: str, tool_description: str, tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult) -> None
```

Update the policy based on the confirmation UI result.

<a id="policies.NeverAskPolicy.to_dict"></a>

#### NeverAskPolicy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

<a id="policies.NeverAskPolicy.from_dict"></a>

#### NeverAskPolicy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy"
```

Deserialize the policy from a dictionary.

<a id="policies.AskOncePolicy"></a>

### AskOncePolicy

Ask only once per tool with specific parameters.

<a id="policies.AskOncePolicy.should_ask"></a>

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

<a id="policies.AskOncePolicy.update_after_confirmation"></a>

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

<a id="policies.AskOncePolicy.to_dict"></a>

#### AskOncePolicy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

<a id="policies.AskOncePolicy.from_dict"></a>

#### AskOncePolicy.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy"
```

Deserialize the policy from a dictionary.

<a id="strategies"></a>

## Module strategies

<a id="strategies.BlockingConfirmationStrategy"></a>

### BlockingConfirmationStrategy

Confirmation strategy that blocks execution to gather user feedback.

<a id="strategies.BlockingConfirmationStrategy.__init__"></a>

#### BlockingConfirmationStrategy.\_\_init\_\_

```python
def __init__(*,
             confirmation_policy: ConfirmationPolicy,
             confirmation_ui: ConfirmationUI,
             reject_template: str = REJECTION_FEEDBACK_TEMPLATE,
             modify_template: str = MODIFICATION_FEEDBACK_TEMPLATE,
             user_feedback_template: str = USER_FEEDBACK_TEMPLATE) -> None
```

Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

**Arguments**:

- `confirmation_policy`: The confirmation policy to determine when to ask for user confirmation.
- `confirmation_ui`: The user interface to interact with the user for confirmation.
- `reject_template`: Template for rejection feedback messages. It should include a `{tool_name}` placeholder.
- `modify_template`: Template for modification feedback messages. It should include `{tool_name}` and `{final_tool_params}`
placeholders.
- `user_feedback_template`: Template for user feedback messages. It should include a `{feedback}` placeholder.

<a id="strategies.BlockingConfirmationStrategy.run"></a>

#### BlockingConfirmationStrategy.run

```python
def run(
    *,
    tool_name: str,
    tool_description: str,
    tool_params: dict[str, Any],
    tool_call_id: str | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None
) -> ToolExecutionDecision
```

Run the human-in-the-loop strategy for a given tool and its parameters.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.
- `tool_call_id`: Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
specific tool invocation.
- `confirmation_strategy_context`: Optional dictionary for passing request-scoped resources. Useful in web/server environments
to provide per-request objects (e.g., WebSocket connections, async queues, Redis pub/sub clients)
that strategies can use for non-blocking user interaction.

**Returns**:

A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
feedback message if rejected.

<a id="strategies.BlockingConfirmationStrategy.run_async"></a>

#### BlockingConfirmationStrategy.run\_async

```python
async def run_async(
    *,
    tool_name: str,
    tool_description: str,
    tool_params: dict[str, Any],
    tool_call_id: str | None = None,
    confirmation_strategy_context: dict[str, Any] | None = None
) -> ToolExecutionDecision
```

Async version of run. Calls the sync run() method by default.

**Arguments**:

- `tool_name`: The name of the tool to be executed.
- `tool_description`: The description of the tool.
- `tool_params`: The parameters to be passed to the tool.
- `tool_call_id`: Optional unique identifier for the tool call.
- `confirmation_strategy_context`: Optional dictionary for passing request-scoped resources.

**Returns**:

A ToolExecutionDecision indicating whether to execute the tool with the given parameters.

<a id="strategies.BlockingConfirmationStrategy.to_dict"></a>

#### BlockingConfirmationStrategy.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the BlockingConfirmationStrategy to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="strategies.BlockingConfirmationStrategy.from_dict"></a>

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

<a id="user_interfaces"></a>

## Module user\_interfaces

<a id="user_interfaces.RichConsoleUI"></a>

### RichConsoleUI

Rich console interface for user interaction.

<a id="user_interfaces.RichConsoleUI.get_user_confirmation"></a>

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

<a id="user_interfaces.RichConsoleUI.to_dict"></a>

#### RichConsoleUI.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the RichConsoleConfirmationUI to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="user_interfaces.RichConsoleUI.from_dict"></a>

#### RichConsoleUI.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI"
```

Deserialize the ConfirmationUI from a dictionary.

<a id="user_interfaces.SimpleConsoleUI"></a>

### SimpleConsoleUI

Simple console interface using standard input/output.

<a id="user_interfaces.SimpleConsoleUI.get_user_confirmation"></a>

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

<a id="user_interfaces.SimpleConsoleUI.to_dict"></a>

#### SimpleConsoleUI.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize the UI to a dictionary.

<a id="user_interfaces.SimpleConsoleUI.from_dict"></a>

#### SimpleConsoleUI.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI"
```

Deserialize the ConfirmationUI from a dictionary.

