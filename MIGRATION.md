# Migration Guide

This document is meant to provide a guide for migrating from Haystack v2.X to v3.0.

---

## How to Document a Breaking Change

When you merge a breaking change into the v3 branch, add an entry to this file under the appropriate section below.
Follow this structure:

### Entry template

```markdown
### <Short title describing what changed>

**What changed:** One or two sentences describing the change — what was removed, renamed, or altered.

**Why:** Brief motivation (e.g. simplification, API consistency, dependency reduction).

**How to migrate:**

Before (v2.x):
\`\`\`python
# example using the old API
from haystack.components.foo import OldComponent
component = OldComponent(old_param="value")
\`\`\`

After (v3.0):
\`\`\`python
# example using the new API
from haystack.components.foo import NewComponent
component = NewComponent(new_param="value")
\`\`\`
```

### Tips

- **One entry per breaking change.** Don't bundle unrelated changes into a single entry.
- **Include a working code example** for every rename, removal, or signature change.
- **Link to the PR** when extra context would help (e.g. `See [#1234](https://github.com/deepset-ai/haystack/pull/1234)`).

---

## Breaking Changes

<!-- Add entries here as v3 development progresses. Example below shows the expected format. -->

### Example entry: `Document.dataframe` field removed

**What changed:** The `dataframe` field on `Document` and the `ExtractedTableAnswer` dataclass have been removed. `pandas` is no longer a required dependency.

**Why:** Reduces the default installation footprint. Components that need `pandas` will raise an informative error prompting the user to install it explicitly.

**How to migrate:**

Before (v2.x):
```python
from haystack.dataclasses import Document
import pandas as pd

doc = Document(content=pd.DataFrame({"col": [1, 2, 3]}))
```

After (v3.0):
```python
# Store tabular data as plain content or create a custom component that returns pandas DataFrames as needed.
from haystack.dataclasses import Document

doc = Document(content="col\n1\n2\n3")
```

### Agent breakpoints removed

**What changed:** The agent-specific breakpoint API has been removed. The `AgentBreakpoint`, `ToolBreakpoint`, and `AgentSnapshot` dataclasses are no longer exported from `haystack.dataclasses`, and the `break_point`, `snapshot`, and `snapshot_callback` parameters have been removed from `Agent.run` and `Agent.run_async`. `Pipeline.run` no longer accepts an `AgentBreakpoint` for its `break_point` argument, and the `agent_snapshot` field has been removed from `PipelineSnapshot`. Pausing and resuming execution inside an Agent (at the chat generator or tool invoker) is no longer supported.

**Why:** Simplifies the breakpoint and snapshot machinery and removes the special-cased agent-internal control flow. Pipeline-level breakpoints still cover the common debugging use cases.

**How to migrate:**

Before (v2.x):
```python
from haystack.components.agents import Agent
from haystack.dataclasses import AgentBreakpoint, Breakpoint, ToolBreakpoint

agent = Agent(chat_generator=..., tools=[...])

# Pause before the chat generator runs
chat_break_point = AgentBreakpoint(
    agent_name="agent",
    break_point=Breakpoint(component_name="chat_generator", visit_count=0),
)

# Or pause before a specific tool is invoked
tool_break_point = AgentBreakpoint(
    agent_name="agent",
    break_point=ToolBreakpoint(component_name="tool_invoker", tool_name="my_tool"),
)

agent.run(messages=[...], break_point=chat_break_point)
```

After (v3.0):
```python
# Pausing inside an Agent is no longer supported. To inspect an Agent's behavior,
# use tracing instead: https://docs.haystack.deepset.ai/docs/tracing
```
