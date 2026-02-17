---
title: "Human-in-the-Loop"
id: human-in-the-loop-api
description: "Abstractions for integrating human feedback and interaction into Agent workflows."
slug: "/human-in-the-loop-api"
---


## `ConfirmationUIResult`

Result of the confirmation UI interaction.

**Parameters:**

- **action** (<code>str</code>) – The action taken by the user such as "confirm", "reject", or "modify".
  This action type is not enforced to allow for custom actions to be implemented.
- **feedback** (<code>str | None</code>) – Optional feedback message from the user. For example, if the user rejects the tool execution,
  they might provide a reason for the rejection.
- **new_tool_params** (<code>dict\[str, Any\] | None</code>) – Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
  they can provide a new set of parameters here.

## `ToolExecutionDecision`

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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Convert the ToolExecutionDecision to a dictionary representation.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary containing the tool execution decision details.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ToolExecutionDecision
```

Populate the ToolExecutionDecision from a dictionary representation.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – A dictionary containing the tool execution decision details.

**Returns:**

- <code>ToolExecutionDecision</code> – An instance of ToolExecutionDecision.

## `AlwaysAskPolicy`

Bases: <code>ConfirmationPolicy</code>

Always ask for confirmation.

### `should_ask`

```python
should_ask(tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool
```

Always ask for confirmation before executing the tool.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – Always returns True, indicating confirmation is needed.

### `update_after_confirmation`

```python
update_after_confirmation(tool_name: str, tool_description: str, tool_params: dict[str, Any], confirmation_result: ConfirmationUIResult) -> None
```

Update the policy based on the confirmation UI result.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ConfirmationPolicy
```

Deserialize the policy from a dictionary.

## `NeverAskPolicy`

Bases: <code>ConfirmationPolicy</code>

Never ask for confirmation.

### `should_ask`

```python
should_ask(tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool
```

Never ask for confirmation, always proceed with tool execution.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – Always returns False, indicating no confirmation is needed.

### `update_after_confirmation`

```python
update_after_confirmation(tool_name: str, tool_description: str, tool_params: dict[str, Any], confirmation_result: ConfirmationUIResult) -> None
```

Update the policy based on the confirmation UI result.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ConfirmationPolicy
```

Deserialize the policy from a dictionary.

## `AskOncePolicy`

Bases: <code>ConfirmationPolicy</code>

Ask only once per tool with specific parameters.

### `should_ask`

```python
should_ask(tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool
```

Ask for confirmation only once per tool with specific parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>bool</code> – True if confirmation is needed, False if already asked with the same parameters.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the policy to a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ConfirmationPolicy
```

Deserialize the policy from a dictionary.

### `update_after_confirmation`

```python
update_after_confirmation(tool_name: str, tool_description: str, tool_params: dict[str, Any], confirmation_result: ConfirmationUIResult) -> None
```

Store the tool and parameters if the action was "confirm" to avoid asking again.

This method updates the internal state to remember that the user has already confirmed the execution of the
tool with the given parameters.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool that was executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters that were passed to the tool.
- **confirmation_result** (<code>ConfirmationUIResult</code>) – The result from the confirmation UI.

## `BlockingConfirmationStrategy`

Confirmation strategy that blocks execution to gather user feedback.

### `__init__`

```python
__init__(*, confirmation_policy: ConfirmationPolicy, confirmation_ui: ConfirmationUI, reject_template: str = REJECTION_FEEDBACK_TEMPLATE, modify_template: str = MODIFICATION_FEEDBACK_TEMPLATE, user_feedback_template: str = USER_FEEDBACK_TEMPLATE) -> None
```

Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

**Parameters:**

- **confirmation_policy** (<code>ConfirmationPolicy</code>) – The confirmation policy to determine when to ask for user confirmation.
- **confirmation_ui** (<code>ConfirmationUI</code>) – The user interface to interact with the user for confirmation.
- **reject_template** (<code>str</code>) – Template for rejection feedback messages. It should include a `{tool_name}` placeholder.
- **modify_template** (<code>str</code>) – Template for modification feedback messages. It should include `{tool_name}` and `{final_tool_params}`
  placeholders.
- **user_feedback_template** (<code>str</code>) – Template for user feedback messages. It should include a `{feedback}` placeholder.

### `run`

```python
run(*, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_call_id: str | None = None, confirmation_strategy_context: dict[str, Any] | None = None) -> ToolExecutionDecision
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

### `run_async`

```python
run_async(*, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_call_id: str | None = None, confirmation_strategy_context: dict[str, Any] | None = None) -> ToolExecutionDecision
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

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the BlockingConfirmationStrategy to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> BlockingConfirmationStrategy
```

Deserializes the BlockingConfirmationStrategy from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>BlockingConfirmationStrategy</code> – Deserialized BlockingConfirmationStrategy.

## `RichConsoleUI`

Bases: <code>ConfirmationUI</code>

Rich console interface for user interaction.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ConfirmationUI
```

Deserialize the ConfirmationUI from a dictionary.

### `get_user_confirmation`

```python
get_user_confirmation(tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> ConfirmationUIResult
```

Get user confirmation for tool execution via rich console prompts.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.

**Returns:**

- <code>ConfirmationUIResult</code> – ConfirmationUIResult based on user input.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serializes the RichConsoleConfirmationUI to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

## `SimpleConsoleUI`

Bases: <code>ConfirmationUI</code>

Simple console interface using standard input/output.

### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize the UI to a dictionary.

### `from_dict`

```python
from_dict(data: dict[str, Any]) -> ConfirmationUI
```

Deserialize the ConfirmationUI from a dictionary.

### `get_user_confirmation`

```python
get_user_confirmation(tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> ConfirmationUIResult
```

Get user confirmation for tool execution via simple console prompts.

**Parameters:**

- **tool_name** (<code>str</code>) – The name of the tool to be executed.
- **tool_description** (<code>str</code>) – The description of the tool.
- **tool_params** (<code>dict\[str, Any\]</code>) – The parameters to be passed to the tool.
