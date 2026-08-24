---
title: "E2B"
id: integrations-e2b
description: "E2B integration for Haystack"
slug: "/integrations-e2b"
---


## haystack_integrations.tools.e2b.bash_tool

### RunBashCommandTool

Bases: <code>Tool</code>

A :class:`~haystack.tools.Tool` that executes bash commands inside an E2B sandbox.

Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
all operate in the same live sandbox environment.

### Usage example

```python
from haystack_integrations.tools.e2b import E2BSandbox, RunBashCommandTool, ReadFileTool

sandbox = E2BSandbox()
agent = Agent(
    chat_generator=...,
    tools=[
        RunBashCommandTool(sandbox=sandbox),
        ReadFileTool(sandbox=sandbox),
    ],
)
```

#### __init__

```python
__init__(sandbox: E2BSandbox) -> None
```

Create a RunBashCommandTool.

**Parameters:**

- **sandbox** (<code>E2BSandbox</code>) – The :class:`E2BSandbox` instance that will execute commands.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RunBashCommandTool
```

Deserialize a RunBashCommandTool from a dictionary.

## haystack_integrations.tools.e2b.e2b_sandbox

### E2BSandbox

Manages the lifecycle of an E2B cloud sandbox.

Instantiate this class and pass it to one or more E2B tool classes
(`RunBashCommandTool`, `ReadFileTool`, `WriteFileTool`,
`ListDirectoryTool`) to share a single sandbox environment across all
tools. All tools that receive the same `E2BSandbox` instance operate
inside the same live sandbox process.

### Usage example

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent

from haystack_integrations.tools.e2b import (
    E2BSandbox,
    RunBashCommandTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

sandbox = E2BSandbox()
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=[
        RunBashCommandTool(sandbox=sandbox),
        ReadFileTool(sandbox=sandbox),
        WriteFileTool(sandbox=sandbox),
        ListDirectoryTool(sandbox=sandbox),
    ],
)
```

Lifecycle is handled automatically by the Agent's pipeline. If you use the
tools standalone, call :meth:`warm_up` before the first tool invocation:

```python
sandbox.warm_up()
# ... use tools ...
sandbox.close()
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("E2B_API_KEY", strict=True),
    sandbox_template: str = "base",
    timeout: int = 120,
    environment_vars: dict[str, str] | None = None,
    instance_id: str | None = None,
) -> None
```

Create an E2BSandbox instance.

**Parameters:**

- **api_key** (<code>Secret</code>) – E2B API key.
- **sandbox_template** (<code>str</code>) – E2B sandbox template name.
- **timeout** (<code>int</code>) – Sandbox inactivity timeout in seconds.
- **environment_vars** (<code>dict\[str, str\] | None</code>) – Optional environment variables to inject into the sandbox.
- **instance_id** (<code>str | None</code>) – Stable identifier preserved across `to_dict`/`from_dict`. When
  omitted, a fresh UUID is generated. Tools that share the same `E2BSandbox`
  instance inherit this id, which is what lets them re-share the instance after
  a serialization round-trip. Distinct from the cloud-side sandbox id assigned
  by E2B at warm-up.

#### warm_up

```python
warm_up() -> None
```

Establish the connection to the E2B sandbox.

Idempotent -- calling it multiple times has no effect if the sandbox is
already running.

**Raises:**

- <code>RuntimeError</code> – If the E2B sandbox cannot be created.

#### close

```python
close() -> None
```

Shut down the E2B sandbox and release all associated resources.

Call this when you are done to avoid leaving idle sandboxes running.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the sandbox configuration to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary containing the serialised configuration.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> E2BSandbox
```

Deserialize an :class:`E2BSandbox` from a dictionary.

Multiple tools that shared a single :class:`E2BSandbox` before serialization
will share the same restored instance: each tool's `from_dict` consults a
process-wide cache keyed on `instance_id`. A cache hit is only honored when
the full serialized config (api_key, template, timeout, environment_vars)
matches the cached entry — a crafted YAML with a guessed id but a different
config falls through to a fresh instance and never observes the cached one.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary created by :meth:`to_dict`.

**Returns:**

- <code>E2BSandbox</code> – An :class:`E2BSandbox` instance ready to be warmed up. May be a
  previously-restored instance if the id and config match.

## haystack_integrations.tools.e2b.list_directory_tool

### ListDirectoryTool

Bases: <code>Tool</code>

A :class:`~haystack.tools.Tool` that lists directory contents in an E2B sandbox.

Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
all operate in the same live sandbox environment.

### Usage example

```python
from haystack_integrations.tools.e2b import E2BSandbox, ListDirectoryTool

sandbox = E2BSandbox()
agent = Agent(chat_generator=..., tools=[ListDirectoryTool(sandbox=sandbox)])
```

#### __init__

```python
__init__(sandbox: E2BSandbox) -> None
```

Create a ListDirectoryTool.

**Parameters:**

- **sandbox** (<code>E2BSandbox</code>) – The :class:`E2BSandbox` instance to list directories from.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ListDirectoryTool
```

Deserialize a ListDirectoryTool from a dictionary.

## haystack_integrations.tools.e2b.read_file_tool

### ReadFileTool

Bases: <code>Tool</code>

A :class:`~haystack.tools.Tool` that reads files from an E2B sandbox filesystem.

Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
all operate in the same live sandbox environment.

### Usage example

```python
from haystack_integrations.tools.e2b import E2BSandbox, ReadFileTool

sandbox = E2BSandbox()
agent = Agent(chat_generator=..., tools=[ReadFileTool(sandbox=sandbox)])
```

#### __init__

```python
__init__(sandbox: E2BSandbox) -> None
```

Create a ReadFileTool.

**Parameters:**

- **sandbox** (<code>E2BSandbox</code>) – The :class:`E2BSandbox` instance to read files from.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ReadFileTool
```

Deserialize a ReadFileTool from a dictionary.

## haystack_integrations.tools.e2b.sandbox_toolset

### E2BToolset

Bases: <code>Toolset</code>

A :class:`~haystack.tools.Toolset` that bundles all E2B sandbox tools.

All tools in the set share a single :class:`E2BSandbox` instance so they
operate inside the same live sandbox process. The toolset owns the sandbox
lifecycle: calling :meth:`warm_up` starts the sandbox, and serialisation
round-trips preserve the shared-sandbox relationship.

### Usage example

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent

from haystack_integrations.tools.e2b import E2BToolset

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o"),
    tools=E2BToolset(),
)
```

#### __init__

```python
__init__(
    api_key: Secret = Secret.from_env_var("E2B_API_KEY", strict=True),
    sandbox_template: str = "base",
    timeout: int = 120,
    environment_vars: dict[str, str] | None = None,
) -> None
```

Create an E2BToolset.

**Parameters:**

- **api_key** (<code>Secret</code>) – E2B API key. Defaults to `Secret.from_env_var("E2B_API_KEY")`.
- **sandbox_template** (<code>str</code>) – E2B sandbox template name. Defaults to `"base"`.
- **timeout** (<code>int</code>) – Sandbox inactivity timeout in seconds. Defaults to `120`.
- **environment_vars** (<code>dict\[str, str\] | None</code>) – Optional environment variables to inject into the sandbox.

#### warm_up

```python
warm_up() -> None
```

Start the shared E2B sandbox (idempotent).

#### close

```python
close() -> None
```

Shut down the shared E2B sandbox and release cloud resources.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this toolset to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> E2BToolset
```

Deserialize an E2BToolset from a dictionary.

## haystack_integrations.tools.e2b.write_file_tool

### WriteFileTool

Bases: <code>Tool</code>

A :class:`~haystack.tools.Tool` that writes files to an E2B sandbox filesystem.

Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
all operate in the same live sandbox environment.

### Usage example

```python
from haystack_integrations.tools.e2b import E2BSandbox, WriteFileTool

sandbox = E2BSandbox()
agent = Agent(chat_generator=..., tools=[WriteFileTool(sandbox=sandbox)])
```

#### __init__

```python
__init__(sandbox: E2BSandbox) -> None
```

Create a WriteFileTool.

**Parameters:**

- **sandbox** (<code>E2BSandbox</code>) – The :class:`E2BSandbox` instance to write files to.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this tool to a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> WriteFileTool
```

Deserialize a WriteFileTool from a dictionary.
