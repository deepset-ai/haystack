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

### ToolInvoker component removed

**What changed:** The `ToolInvoker` component has been removed. Imports from `haystack.components.tools`
and pipelines that use `ToolInvoker` as a standalone component are no longer supported.

**Why:** Tool execution is now owned by `Agent`, so the tool-calling loop, state handling, streaming callback
passthrough, warm-up, and sync/async execution live in one place.

**How to migrate:** Pass tools directly to `Agent` instead of wiring a chat generator to `ToolInvoker`.
The `Agent` will pass tool definitions to the chat generator, execute requested tool calls, append tool
results to the conversation, and continue the loop until an exit condition is reached.

Before (v2.x):
```python
from typing import Annotated

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import tool


@tool
def weather(city: Annotated[str, "The name of the city"]) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", tools=[weather])
tool_invoker = ToolInvoker(tools=[weather])

llm_result = chat_generator.run(messages=[ChatMessage.from_user("What is the weather in Berlin?")])
tool_result = tool_invoker.run(messages=llm_result["replies"])
```

After (v3.0):
```python
from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool


@tool
def weather(city: Annotated[str, "The name of the city"]) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[weather])
result = agent.run(messages=[ChatMessage.from_user("What is the weather in Berlin?")])
```

The `tool_invoker_kwargs` parameter has been removed from `Agent`. Previously, `ToolInvoker` options were
forwarded through this dictionary; the relevant options are now top-level `Agent` constructor parameters:

- `max_workers` is now the top-level `tool_concurrency_limit` parameter.
- `enable_streaming_callback_passthrough` is now the top-level `tool_streaming_callback_passthrough` parameter.

Before (v2.x):
```python
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[weather],
    tool_invoker_kwargs={"max_workers": 4, "enable_streaming_callback_passthrough": True},
)
```

After (v3.0):
```python
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[weather],
    tool_concurrency_limit=4,
    tool_streaming_callback_passthrough=True,
)
```

The `convert_result_to_json_string` option (also previously set through `tool_invoker_kwargs`) has been removed.
Non-string tool results are now always serialized with `json.dumps` rather than `str`, which changes their string form.

### Agent

#### Breakpoint and snapshot API removed

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
# use tracing instead (https://docs.haystack.deepset.ai/docs/tracing). For example,
# you can wire up a Langfuse tracer for a standalone Agent by instantiating a
# LangfuseConnector — its constructor registers the tracer globally, so any
# subsequent Agent.run call will be traced.

# NOTE: install the langfuse integration first with `pip install langfuse-haystack` to run this example.
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

# Instantiating the connector enables tracing globally — no need to add it to a pipeline.
LangfuseConnector("Standalone Agent example")

agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[...])
agent.run(messages=[ChatMessage.from_user("What's the weather in Berlin?")])
```

#### Tracing span hierarchy reshaped

**What changed:** Each iteration of the Agent loop now emits a single `haystack.agent.step` Haystack span with two nested children — `haystack.agent.step.llm` for the chat generator call and `haystack.agent.step.tool` for the tool invocation (only when tool calls happen). Previously each iteration produced two child spans through `Pipeline._run_component` (one for the chat generator, one for the tool invoker) tagged with `haystack.component.name` / `haystack.component.type`. The new spans do NOT carry `haystack.component.*` tags; they expose new content tags `haystack.agent.step.llm.input`/`.output` and `haystack.agent.step.tool.input`/`.output`.

**Why:** Removes the dependency on `Pipeline._run_component` inside `Agent.run` and produces a clearer per-iteration trace structure that maps directly onto common agent-tracing conventions (e.g., Langfuse `chain → {generation, tool}`).

**How to migrate:** Custom tracer backends or `SpanHandler` implementations that dispatched on `component_type == "ToolInvoker"` or `component_type.endswith("ChatGenerator")` for Agent-internal spans must now dispatch on the new operation names instead. For example, in the Langfuse integration this means subclassing or updating the `DefaultSpanHandler` and overriding `create_span` to recognize the three new operations:

```python
from typing import cast
from haystack_integrations.tracing.langfuse import DefaultSpanHandler, LangfuseSpan
from haystack_integrations.tracing.langfuse.tracer import ObservationSpanType, SpanContext


class AgentStepSpanHandler(DefaultSpanHandler):
    def create_span(self, context: SpanContext) -> LangfuseSpan:
        if context.operation_name == "haystack.agent.step":
            return LangfuseSpan(
                self.tracer.start_as_current_observation(
                    name=f"agent step {context.tags.get('haystack.agent.step', 0)}",
                    as_type=cast(ObservationSpanType, "chain"),
                )
            )
        if context.operation_name == "haystack.agent.step.llm":
            return LangfuseSpan(
                self.tracer.start_as_current_observation(name="llm", as_type=cast(ObservationSpanType, "generation"))
            )
        if context.operation_name == "haystack.agent.step.tool":
            return LangfuseSpan(
                self.tracer.start_as_current_observation(name="tool", as_type=cast(ObservationSpanType, "tool"))
            )
        return super().create_span(context)
```

Pass an instance to the `LangfuseConnector`:

```python
LangfuseConnector("My Agent", span_handler=AgentStepSpanHandler())
```

#### Runtime `user_prompt` and `system_prompt` removed from `Agent.run` / `Agent.run_async`

**What changed:** The `user_prompt` and `system_prompt` parameters have been removed from `Agent.run` and `Agent.run_async`. Both prompts must now be set at initialization time on the `Agent`; they can no longer be passed per-run, including via `Pipeline.run(data={"agent": {...}})`.

**Why:** A single source of truth for the Agent's prompt templates simplifies the API and makes prompts a stable part of the Agent configuration. It also removes the old requirement that variables used by a runtime `user_prompt` override had to already exist in the initialization-time `user_prompt` for pipeline input sockets to be created.

**How to migrate:**

Before (v2.x), overriding prompts through `Pipeline.run`:
```python
from haystack import Pipeline
from haystack.components.agents import Agent

init_user_prompt = (
    "{% message role='user' %}"
    "Answer {{query}} using these documents:"
    "{% for doc in documents %}{{doc.content}}{% endfor %}"
    "{% endmessage %}"
)
agent = Agent(
    chat_generator=...,
    tools=[...],
    system_prompt="Default system prompt.",
    user_prompt=init_user_prompt,
)
pipeline = Pipeline()
pipeline.add_component("retriever", ...)
pipeline.add_component("agent", agent)
pipeline.connect("retriever.documents", "agent.documents")

pipeline.run(
    data={
        "retriever": {"query": query},
        "agent": {
            "messages": [],
            "query": query,
            "system_prompt": "You are a knowledgeable assistant.",
            "user_prompt": (
                "{% message role='user' %}"
                "Use these documents to answer {{query}}:"
                "{% for doc in documents %}{{doc.content}}{% endfor %}"
                "{% endmessage %}"
            ),
        },
    }
)
```

After (v3.0), configure both prompts on the `Agent`:
```python
from haystack import Pipeline
from haystack.components.agents import Agent

agent = Agent(
    chat_generator=...,
    tools=[...],
    system_prompt="You are a knowledgeable assistant.",
    # `user_prompt` can be a plain string template or an explicit Jinja2 message template.
    # Explicit message templates must contain a single message block that renders as a user message.
    user_prompt="Use these documents to answer {{query}}: {% for doc in documents %}{{doc.content}}{% endfor %}",
)
pipeline = Pipeline()
pipeline.add_component("retriever", ...)
pipeline.add_component("agent", agent)
pipeline.connect("retriever.documents", "agent.documents")

pipeline.run(data={"retriever": {"query": query}, "agent": {"messages": [], "query": query}})
```

If the prompt itself must still be assembled per run, build `ChatMessage` objects before the `Agent` (e.g. with a `ChatPromptBuilder`) and pass them through the `messages` input.
For a runtime system prompt, construct an `Agent` without `system_prompt` or `user_prompt` and include a system message at the start of `messages`.

#### Reserved `state_schema` keys for built-in run metadata

**What changed:** `Agent` now auto-populates three new outputs — `step_count`, `token_usage`, and `tool_call_counts` — and reserves those names in its `state_schema`. Passing any of them as a `state_schema` key now raises `ValueError`.

**Why:** These keys are managed by `Agent` itself and exposed as outputs only; allowing users to redefine them would let an input shadow the value the Agent is trying to write.

**How to migrate:** Rename any clashing `state_schema` entries.

Before (v2.x):
```python
agent = Agent(
    chat_generator=...,
    tools=[...],
    state_schema={"token_usage": {"type": dict}},
)
```

After (v3.0):
```python
agent = Agent(
    chat_generator=...,
    tools=[...],
    state_schema={"my_token_usage": {"type": dict}},
)
```

### LLM

#### Runtime `user_prompt` and `system_prompt` removed from `LLM.run` / `LLM.run_async`

**What changed:** The `user_prompt` and `system_prompt` parameters have been removed from `LLM.run` and `LLM.run_async`. Both prompts must now be set at initialization time on the `LLM`; they can no longer be passed per-run, including via `Pipeline.run(data={"llm": {...}})`.

**Why:** `LLM` prompt templates define the component's dynamic input sockets. The old runtime override path was misleading because any variables used by the override still had to be present in the initialization-time `user_prompt` for pipeline connections such as `llm.documents` to exist.

**How to migrate:**

Before (v2.x), overriding prompts through `Pipeline.run`:
```python
from haystack import Pipeline
from haystack.components.generators.chat import LLM

init_user_prompt = "Answer {{query}} using these documents: {% for doc in documents %}{{doc.content}}{% endfor %}"
llm = LLM(
    chat_generator=...,
    system_prompt="Default system prompt.",
    user_prompt=init_user_prompt,
)
pipeline = Pipeline()
pipeline.add_component("retriever", ...)
pipeline.add_component("llm", llm)

# This connection only worked because `documents` was already present in
# `init_user_prompt`, so the LLM had a `documents` input socket.
pipeline.connect("retriever.documents", "llm.documents")

pipeline.run(
    data={
        "retriever": {"query": query},
        "llm": {
            "query": query,
            "system_prompt": "You are a knowledgeable assistant.",
            "user_prompt": "Use these documents to answer {{query}}: {% for doc in documents %}{{doc.content}}{% endfor %}",
        },
    }
)
```

After (v3.0), configure both prompts on the `LLM`:
```python
from haystack import Pipeline
from haystack.components.generators.chat import LLM

llm = LLM(
    chat_generator=...,
    system_prompt="You are a knowledgeable assistant.",
    user_prompt="Use these documents to answer {{query}}: {% for doc in documents %}{{doc.content}}{% endfor %}",
)
pipeline = Pipeline()
pipeline.add_component("retriever", ...)
pipeline.add_component("llm", llm)
pipeline.connect("retriever.documents", "llm.documents")

pipeline.run(data={"retriever": {"query": query}, "llm": {"query": query}})
```

If the prompt itself must still be assembled per run, build `ChatMessage` objects before the `LLM` and pass them through the `messages` input. For a runtime system prompt, construct an `LLM` without `system_prompt` or `user_prompt` and include a system message at the start of `messages`.

### `PromptBuilder` and `ChatPromptBuilder` template variables are required by default

**What changed:** `PromptBuilder` and `ChatPromptBuilder` now treat every Jinja2 template variable as required by default. Previously, variables were optional by default and missing values were silently rendered as empty strings. The `required_variables` parameter's default has been changed from `None` (all optional) to `"*"` (all required). Passing `required_variables=None` explicitly still opts into the old "all optional" behavior.

**Why:** Avoids silent rendering bugs where a missing variable produces an unexpectedly empty section of the prompt — especially in multi-branch pipelines where the issue often surfaces far from its root cause. Aligns the default with `ConditionalRouter`'s convention that inputs are required unless declared otherwise.

**How to migrate:**

Before (v2.x):
```python
from haystack.components.builders import PromptBuilder

# All variables were optional by default; missing values rendered as "".
builder = PromptBuilder(template="Hello, {{ name }}! {{ greeting }}")
builder.run(name="John")  # greeting silently becomes "" → "Hello, John! "
```

After (v3.0):
```python
from haystack.components.builders import PromptBuilder

# Option 1: provide every variable (matches the new safe default).
builder = PromptBuilder(template="Hello, {{ name }}! {{ greeting }}")
builder.run(name="John", greeting="Welcome")

# Option 2: declare which variables are required; everything else stays optional.
builder = PromptBuilder(
    template="Hello, {{ name }}! {{ greeting }}",
    required_variables=["name"],
)
builder.run(name="John")  # greeting renders as ""

# Option 3: restore the old "all optional" behavior.
builder = PromptBuilder(
    template="Hello, {{ name }}! {{ greeting }}",
    required_variables=None,
)
builder.run(name="John")  # greeting renders as ""
```
