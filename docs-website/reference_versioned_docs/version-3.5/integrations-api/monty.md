---
title: "Monty"
id: integrations-monty
description: "Monty integration for Haystack"
slug: "/integrations-monty"
---


## haystack_integrations.tools.monty.python_tool

### MontyPythonTool

Bases: <code>Tool</code>

A Haystack `Tool` that lets an `Agent` run Python code in a [Monty](https://pydantic.dev/docs/monty/) sandbox.

Monty is a minimal Python interpreter written in Rust. The tool keeps a pool of Monty worker processes and runs
every call in a fresh interpreter, so nothing leaks between calls, users, or concurrent tool invocations. The
tool returns what the code printed, the `repr()` of its last expression, and any error as a traceback, so the
LLM can read the result and fix its code.

### Security model

Monty is a language-level sandbox: its interpreter implements no operation that reaches the host, so code has
no access to files, the network, environment variables, or subprocesses. It runs in worker subprocesses started
with an empty environment, so a crash never takes down the host process. The tool mounts no directories and
exposes no host functions to the sandbox. Execution time and heap memory are capped by `resource_limits`.

### Usage example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.monty import MontyPythonTool

# Requires the OPENAI_API_KEY environment variable
agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[MontyPythonTool()])
result = agent.run(messages=[ChatMessage.from_user("What is the sum of the first 100 prime numbers?")])
print(result["last_message"].text)
# >> The sum of the first 100 prime numbers is 24133.
```

#### __init__

```python
__init__(
    *,
    name: str = "run_python",
    description: str | None = None,
    resource_limits: ResourceLimits | None = None,
    type_check: bool = False,
    max_output_chars: int = 20000
) -> None
```

Create a MontyPythonTool.

**Parameters:**

- **name** (<code>str</code>) – Tool name exposed to the LLM.
- **description** (<code>str | None</code>) – Tool description exposed to the LLM. If `None`, a description of the sandbox and the
  supported Python subset is used.
- **resource_limits** (<code>ResourceLimits | None</code>) – Monty resource limits for each call, merged over the defaults of 30 seconds of
  execution time (`max_feed_duration_secs`) and 256 MiB of heap memory (`max_memory`). Set a key to `None`
  to disable that limit. See `pydantic_monty.ResourceLimits` for the available keys.
- **type_check** (<code>bool</code>) – If `True`, type-check the code with Monty's bundled type checker before running it, and
  return type errors to the LLM instead of executing the code.
- **max_output_chars** (<code>int</code>) – Maximum number of characters kept from each of the printed output, the result, and
  the error before they are returned to the LLM. The output and the result keep their beginning, the error
  keeps its end, where the exception is.

**Raises:**

- <code>ValueError</code> – If `resource_limits` contains a key that Monty doesn't support, or `max_output_chars` is
  less than 1.

#### warm_up

```python
warm_up() -> None
```

Start the pool of Monty worker processes. Called by `Agent.warm_up()`; safe to call more than once.

#### close

```python
close() -> None
```

Shut down the pool of Monty worker processes.

Safe to call more than once. The tool starts a new pool if it is invoked again.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MontyPythonTool
```

Deserialize the tool from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>MontyPythonTool</code> – Deserialized tool.
