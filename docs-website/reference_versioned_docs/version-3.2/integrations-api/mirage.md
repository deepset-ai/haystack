---
title: "Mirage"
id: integrations-mirage
description: "Mirage integration for Haystack"
slug: "/integrations-mirage"
---


## haystack_integrations.tools.mirage.shell_tool

### MirageShellTool

Bases: <code>Tool</code>

A Haystack `Tool` that lets an `Agent` run bash commands across a Mirage virtual filesystem.

Mirage mounts heterogeneous backends (object storage, databases, SaaS apps, local disk) as one
filesystem; this tool exposes Mirage's single `execute` surface to an Agent as one well-described
tool with a `command` parameter. Output is normalized to text and truncated before it reaches the
model.

### Security model

Mirage never shells out to the host: every command runs inside Mirage's own virtual-filesystem
interpreter, so the blast radius is confined to the mounts you attach. Two controls shape what an
Agent can do:

- **Per-mount read-only mode** (`MirageMount(..., read_only=True)`) is the authoritative write
  boundary. Mirage refuses any write to a read-only mount regardless of the command used, so this
  -- not the allowlist -- is how you prevent modification or deletion. Mount anything the Agent
  should not change as read-only.
- **The command allowlist** (`allowed_commands`) restricts *which* commands may run. It is
  enforced against every command Mirage would execute, including commands nested inside
  `$(...)`, backticks, `<(...)` and subshells, so `ls "$(rm x)"` is rejected unless `rm`
  is also allowed. Treat it as a best-effort filter to steer the Agent, not a sandbox: allowing a
  command that itself runs other commands (`eval`, `bash`, `sh`, `source`, `xargs`,
  `timeout`) effectively allows anything, so do not list those for untrusted/hosted use.
- **`denied_paths`** rejects any command whose text references one of the given path substrings.

### Usage example

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mirage import MirageWorkspace, MirageMount, MirageShellTool

workspace = MirageWorkspace([
    MirageMount(path="/data", resource="ram"),
    MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True),
])
tool = MirageShellTool(workspace, allowed_commands=["ls", "cat", "grep", "head", "wc", "cp"])

agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[tool])
result = agent.run(messages=[ChatMessage.from_user("How many lines in /s3/log.txt mention 'alert'?")])
print(result["messages"][-1].text)
```

#### __init__

```python
__init__(
    workspace: MirageWorkspace,
    *,
    name: str = "mirage_shell",
    description: str | None = None,
    invocation_timeout: float = 60.0,
    max_output_chars: int = 20000,
    allowed_commands: list[str] | None = None,
    denied_paths: list[str] | None = None
) -> None
```

Initialize the Mirage shell tool.

**Parameters:**

- **workspace** (<code>MirageWorkspace</code>) – The :class:`MirageWorkspace` describing the mount tree.
- **name** (<code>str</code>) – Tool name exposed to the LLM.
- **description** (<code>str | None</code>) – Custom description. If None, one is generated from the mount tree.
- **invocation_timeout** (<code>float</code>) – Maximum seconds to wait for a command to finish.
- **max_output_chars** (<code>int</code>) – Truncate command output to this many characters before returning it.
- **allowed_commands** (<code>list\[str\] | None</code>) – If set, only these command names may run, e.g.
  `["ls", "cat", "grep", "head", "wc"]`. The allowlist is enforced against *every* command
  Mirage would execute -- including commands nested in substitutions/subshells -- so
  `ls "$(rm x)"` is rejected unless `rm` is also allowed. It is a filter over Mirage's
  virtual commands to steer the Agent, not a security sandbox; the write boundary is
  per-mount `read_only` (see the class "Security model" section). If None, any command is
  allowed (not recommended for untrusted/hosted use).
- **denied_paths** (<code>list\[str\] | None</code>) – If set, any command referencing one of these path substrings is rejected.

#### warm_up

```python
warm_up() -> None
```

Build the underlying live workspace eagerly. Called by `Agent.warm_up()`/`Pipeline.warm_up()`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the tool to a dictionary in the `{"type": ..., "data": ...}` format.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MirageShellTool
```

Deserialize the tool from a dictionary.

#### close

```python
close() -> None
```

Close the underlying workspace.

## haystack_integrations.tools.mirage.workspace

### MirageMount

Declarative description of a single backend mounted into a :class:`MirageWorkspace`.

A mount is the serializable unit of a Mirage workspace: it names *where* a backend is mounted
(`path`), *which* backend it is (`resource`, a Mirage registry name such as `"s3"` or `"gdrive"`),
and *how* to configure it (`config`).

`config` values may be plain values, Haystack `Secret` objects for credentials, or an OAuth token
source (e.g. `OAuthRefreshTokenSource`) for backends whose config accepts a token-provider callable
(such as Mirage's OneDrive `access_token`). Secrets and token sources are resolved only when the
live workspace is built.

Every backend is created the same way. Use the Mirage registry name and the config keys that backend expects
(discover names with `MirageMount.available_resources()`; config keys come from the backend's Mirage config class):

```python
from haystack.utils import Secret

MirageMount(path="/data", resource="ram")                                  # in-memory scratch
MirageMount(path="/local", resource="disk", config={"root": "/srv/data"})  # local disk
MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True)
MirageMount(
    path="/drive",
    resource="gdrive",
    config={"client_id": "...", "refresh_token": Secret.from_env_var("GDRIVE_REFRESH_TOKEN")},
    read_only=True,
)
```

**Parameters:**

- **path** (<code>str</code>) – Mount point in the virtual filesystem, e.g. `"/s3"`.
- **resource** (<code>str</code>) – Mirage registry name of the backend, e.g. `"ram"`, `"disk"`, `"s3"`, `"gdrive"`.
  See `mirage.resource.registry.REGISTRY` or `MirageMount.available_resources()` for the full list.
- **config** (<code>dict\[str, Any\]</code>) – Keyword arguments passed to the backend's Mirage config. Values may be `Secret`s, or
  an OAuth token source that is turned into a token-provider callable when the workspace is built.
- **read_only** (<code>bool</code>) – If True, the mount is mounted in Mirage's READ mode and writes are rejected by
  Mirage itself.

#### available_resources

```python
available_resources() -> list[str]
```

Return the Mirage registry names usable as `resource`.

These are short backend names such as `"s3"`, `"gdrive"`, `"postgres"`. Pass one to
`MirageMount(resource=...)`; the config keys each backend expects come from its Mirage
config class.

### MirageWorkspace

A description of a Mirage mount tree that lazily builds a live `mirage.Workspace`.

`MirageWorkspace` is the shared backend behind the Mirage tools and components: it holds the list of
:class:`MirageMount`s and the cache configuration, serializes cleanly (resolving `Secret`s only at
build time), and constructs the live workspace on first use via Mirage's resource registry.

### Usage example

```python
from haystack.utils import Secret
from haystack_integrations.tools.mirage import MirageWorkspace, MirageMount

ws = MirageWorkspace(
    mounts=[
        MirageMount(path="/data", resource="ram"),
        MirageMount(path="/s3", resource="s3", config={"bucket": "my-bucket"}, read_only=True),
    ]
)
print(ws.run("ls /s3"))
```

#### __init__

```python
__init__(
    mounts: list[MirageMount], *, cache_limit: str | int = "512MB"
) -> None
```

Initialize the workspace description.

**Parameters:**

- **mounts** (<code>list\[MirageMount\]</code>) – The backends to mount, as a list of :class:`MirageMount`.
- **cache_limit** (<code>str | int</code>) – Mirage file-cache size limit (e.g. `"512MB"` or an int byte count).

**Raises:**

- <code>MirageConfigError</code> – If no mounts are provided or mount paths are not unique.

#### warm_up

```python
warm_up() -> None
```

Build the live `mirage.Workspace` eagerly. Idempotent.

#### close

```python
close() -> None
```

Close the live workspace and release its resources, if it was built. Thread-safe.

#### run

```python
run(
    command: str, *, timeout: float = 60.0, max_chars: int | None = None
) -> str
```

Run a bash `command` against the mount tree from a synchronous context and return its output.

**Parameters:**

- **command** (<code>str</code>) – A bash command line, e.g. `"grep -r alert /s3/logs | wc -l"`.
- **timeout** (<code>float</code>) – Maximum seconds to wait for the command.
- **max_chars** (<code>int | None</code>) – If set, truncate the returned text to this many characters.

**Returns:**

- <code>str</code> – Combined stdout (plus a trailing error note on non-zero exit) as a string.

#### run_async

```python
run_async(
    command: str, *, timeout: float = 60.0, max_chars: int | None = None
) -> str
```

Async counterpart of :meth:`run`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize the workspace description to a dictionary (Secret-safe).

#### from_dict

```python
from_dict(data: dict[str, Any]) -> MirageWorkspace
```

Deserialize a workspace description from a dictionary.

#### describe

```python
describe() -> str
```

Return a human/LLM-readable summary of the mount tree (used in tool descriptions).
