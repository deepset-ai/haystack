# Migration Guide

This document is meant to provide a guide for migrating from Haystack v2.X to v3.0.

---

## Breaking Changes

### `GeneratedAnswer` and `ExtractedAnswer` serialization format

**What changed:** `GeneratedAnswer.to_dict()` and `ExtractedAnswer.to_dict()` now return a flat dictionary of the object's fields instead of wrapping them in a `{"type": ..., "init_parameters": {...}}` envelope.

**Why:** Aligns these dataclasses with how every other Haystack dataclass (`Document`, `ChatMessage`, etc.) serializes, and removes redundant type metadata from pipeline snapshots and `State` objects.

**Deserialization is backward compatible:** `from_dict()` accepts both the new flat format and the old wrapped `{"type": ..., "init_parameters": {...}}` format, so existing serialized artifacts (pipeline snapshots, breakpoints, `State` objects) keep loading without any changes on your side.

**How to migrate:** Only code that *reads* the serialized output needs updating: access fields at the top level instead of under `init_parameters`. Code that deserializes with `from_dict()` needs no changes.

Before (v2.x):
```python
serialized = generated_answer.to_dict()
data = serialized["init_parameters"]["data"]
```

After (v3.0):
```python
serialized = generated_answer.to_dict()
data = serialized["data"]

# Deserialization still accepts both the new and the old format:
GeneratedAnswer.from_dict(serialized)          # new flat format
GeneratedAnswer.from_dict(old_wrapped_dict)    # old {"type": ..., "init_parameters": {...}} format
```

### Components Moved to External Packages

**What changed:** Some components have been moved out of Haystack into dedicated integration packages,
hosted in the [haystack-core-integrations](https://github.com/deepset-ai/haystack-core-integrations) repository.

**Why:** Moving these components to separate packages allows testing more thoroughly in isolation and
releasing fixes independently of the Haystack release cycle. This also makes Haystack development and CI leaner.

**How to migrate:** Install the new package and update your imports as shown in the table below.

```bash
pip install <new-package>
```

| Old import (`haystack-ai<3.0.0`) | New package | New import |
|---|---|---|
| `from haystack.components.generators.chat import HuggingFaceAPIChatGenerator` | `huggingface-api-haystack` | `from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator` |
| `from haystack.components.embedders import HuggingFaceAPITextEmbedder` | `huggingface-api-haystack` | `from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPITextEmbedder` |
| `from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder` | `huggingface-api-haystack` | `from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPIDocumentEmbedder` |
| `from haystack.components.rankers import HuggingFaceTEIRanker` | `huggingface-api-haystack` | `from haystack_integrations.components.rankers.huggingface_api import HuggingFaceTEIRanker` |
| `from haystack.components.classifiers import TransformersZeroShotDocumentClassifier` | `transformers-haystack` | `from haystack_integrations.components.classifiers.transformers import TransformersZeroShotDocumentClassifier` |
| `from haystack.components.generators.chat import HuggingFaceLocalChatGenerator` | `transformers-haystack` | `from haystack_integrations.components.generators.transformers import TransformersChatGenerator` |
| `from haystack.components.readers import ExtractiveReader` | `transformers-haystack` | `from haystack_integrations.components.readers.transformers import TransformersExtractiveReader` |
| `from haystack.components.routers import TransformersTextRouter` | `transformers-haystack` | `from haystack_integrations.components.routers.transformers import TransformersTextRouter` |
| `from haystack.components.routers import TransformersZeroShotTextRouter` | `transformers-haystack` | `from haystack_integrations.components.routers.transformers import TransformersZeroShotTextRouter` |
| `from haystack.components.websearch import SerperDevWebSearch` | `serperdev-haystack` | `from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch` |
| `from haystack.components.websearch import SearchApiWebSearch` | `searchapi-haystack` | `from haystack_integrations.components.websearch.searchapi import SearchApiWebSearch` |
| `from haystack.components.classifiers import DocumentLanguageClassifier` | `langdetect-haystack` | `from haystack_integrations.components.classifiers.langdetect import DocumentLanguageClassifier` |
| `from haystack.components.routers import TextLanguageRouter` | `langdetect-haystack` | `from haystack_integrations.components.routers.langdetect import TextLanguageRouter` |
| `from haystack.components.audio import LocalWhisperTranscriber` | `whisper-haystack` | `from haystack_integrations.components.audio.whisper import LocalWhisperTranscriber` |
| `from haystack.components.audio import RemoteWhisperTranscriber` | `whisper-haystack` | `from haystack_integrations.components.audio.whisper import RemoteWhisperTranscriber` |
| `from haystack.components.extractors import NamedEntityExtractor` (Hugging Face backend) | `transformers-haystack` | `from haystack_integrations.components.extractors.transformers import TransformersNamedEntityExtractor` |
| `from haystack.components.extractors import NamedEntityExtractor` (spaCy backend) | `spacy-haystack` | `from haystack_integrations.components.extractors.spacy import SpacyNamedEntityExtractor` |
| `from haystack.components.embedders import SentenceTransformersTextEmbedder` | `sentence-transformers-haystack` | `from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder` |
| `from haystack.components.embedders import SentenceTransformersDocumentEmbedder` | `sentence-transformers-haystack` | `from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder` |
| `from haystack.components.embedders import SentenceTransformersSparseTextEmbedder` | `sentence-transformers-haystack` | `from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersSparseTextEmbedder` |
| `from haystack.components.embedders import SentenceTransformersSparseDocumentEmbedder` | `sentence-transformers-haystack` | `from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersSparseDocumentEmbedder` |
| `from haystack.components.embedders.image import SentenceTransformersDocumentImageEmbedder` | `sentence-transformers-haystack` | `from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentImageEmbedder` |
| `from haystack.components.rankers import SentenceTransformersSimilarityRanker` | `sentence-transformers-haystack` | `from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker` |
| `from haystack.components.rankers import SentenceTransformersDiversityRanker` | `sentence-transformers-haystack` | `from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersDiversityRanker` |
| `from haystack.tracing.datadog import DatadogTracer` | `datadog-haystack` | `from haystack_integrations.tracing.datadog import DatadogTracer` |
| `from haystack.tracing.opentelemetry import OpenTelemetryTracer` | `opentelemetry-haystack` | `from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer` |
| `from haystack.tracing import OpenTelemetryTracer` | `opentelemetry-haystack` | `from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer` |
| `from haystack.components.converters import TikaDocumentConverter` | `tika-haystack` | `from haystack_integrations.components.converters.tika import TikaDocumentConverter` |
| `from haystack.components.converters import AzureOCRDocumentConverter` | `azure-form-recognizer-haystack` | `from haystack_integrations.components.converters.azure_form_recognizer import AzureOCRDocumentConverter` |
| `from haystack.components.connectors import OpenAPIConnector` | `openapi-haystack` | `from haystack_integrations.components.connectors.openapi import OpenAPIConnector` |
| `from haystack.components.connectors import OpenAPIServiceConnector` | `openapi-haystack` | `from haystack_integrations.components.connectors.openapi import OpenAPIServiceConnector` |
| `from haystack.components.converters import OpenAPIServiceToFunctions` | `openapi-haystack` | `from haystack_integrations.components.converters.openapi import OpenAPIServiceToFunctions` |

### `DatadogTracer` moved to the `datadog-haystack` integration

**What changed:** The `DatadogTracer` has been moved out of Haystack into the `datadog-haystack` integration package.
In addition, Haystack no longer automatically enables Datadog tracing when `ddtrace` is installed. You now enable it
explicitly by adding the new `DatadogConnector` component to your pipeline.

**Why:** Moving the tracer to a dedicated package keeps Haystack's dependencies leaner and lets the integration be
released independently. Removing the implicit auto-enable makes tracing setup explicit and predictable.

**How to migrate:**

Install the integration:

```bash
pip install datadog-haystack
```

Before (v2.x), Datadog tracing was auto-enabled when `ddtrace` was installed, or set up manually:

```python
import ddtrace
from haystack import tracing
from haystack.tracing.datadog import DatadogTracer

tracing.enable_tracing(DatadogTracer(ddtrace.tracer))
```

After (v3.0), add the `DatadogConnector` to your pipeline to enable tracing:

```python
from haystack import Pipeline
from haystack_integrations.components.connectors.datadog import DatadogConnector

pipe = Pipeline()
pipe.add_component("tracer", DatadogConnector())
```

Alternatively, you can still enable the tracer manually using the new import path:

```python
import ddtrace
from haystack import tracing
from haystack_integrations.tracing.datadog import DatadogTracer

tracing.enable_tracing(DatadogTracer(ddtrace.tracer))
```

### `OpenTelemetryTracer` moved to the `opentelemetry-haystack` integration

**What changed:** The `OpenTelemetryTracer` has been moved out of Haystack into the `opentelemetry-haystack`
integration package, and the `opentelemetry-sdk` dependency is no longer installed with Haystack. In addition,
Haystack no longer automatically enables OpenTelemetry tracing when `opentelemetry-sdk` is installed and configured.
You now enable it explicitly by adding the new `OpenTelemetryConnector` component to your pipeline.

**Why:** Moving the tracer to a dedicated package keeps Haystack's dependencies leaner and lets the integration be
released independently. Removing the implicit auto-enable makes tracing setup explicit and predictable.

**How to migrate:**

Install the integration:

```bash
pip install opentelemetry-haystack
```

Before (v2.x), OpenTelemetry tracing was auto-enabled when `opentelemetry-sdk` was installed and configured, or set
up manually:

```python
from opentelemetry import trace
from haystack import tracing
from haystack.tracing import OpenTelemetryTracer

tracing.enable_tracing(OpenTelemetryTracer(trace.get_tracer("my_application")))
```

After (v3.0), enable the tracer manually using the new import path:

```python
from opentelemetry import trace
from haystack import tracing
from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer

tracing.enable_tracing(OpenTelemetryTracer(trace.get_tracer("my_application")))
```

Alternatively, add the `OpenTelemetryConnector` to your pipeline to enable tracing:

```python
from haystack import Pipeline
from haystack_integrations.components.connectors.opentelemetry import OpenTelemetryConnector

pipe = Pipeline()
pipe.add_component("tracer", OpenTelemetryConnector())
```

**Also removed:** `haystack.tracing.auto_enable_tracing` (it is no longer called on `import haystack`). Because
Haystack no longer ships a built-in tracing backend, there is nothing to auto-enable. Enable tracing explicitly via
a connector (such as `OpenTelemetryConnector`) or with `haystack.tracing.enable_tracing(...)`. The
`HAYSTACK_AUTO_TRACE_ENABLED` environment variable no longer has any effect.

### `TransformersSimilarityRanker` removed

**What changed:** The `TransformersSimilarityRanker` component has been removed. It was not moved to an
integration package.

**Why:** The component was in legacy state and no longer received updates. `SentenceTransformersSimilarityRanker`
provides the same functionality plus async support and the more capable sentence-transformers backend.

**How to migrate:** Use `SentenceTransformersSimilarityRanker`, which accepts the same parameters.

Before (v2.x):
```python
from haystack.components.rankers import TransformersSimilarityRanker

ranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

After (v3.0):
```python
from haystack.components.rankers import SentenceTransformersSimilarityRanker

ranker = SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
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

### `haystack-experimental` is no longer a core dependency

**What changed:** `haystack-experimental` has been removed from Haystack's core dependencies. Installing `haystack-ai` no longer pulls in `haystack-experimental` automatically.

**Why:** Reduces the default installation footprint. Experimental features are opt-in and should not be installed for users who do not need them.

**How to migrate:** If your code imports from `haystack_experimental` (directly or through an integration that depends on it), install the package explicitly:

```bash
pip install haystack-experimental
```

Installations that do not use `haystack_experimental` require no changes.

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

**What changed:** Each iteration of the Agent loop now emits a single `haystack.agent.step` Haystack span with nested children: `haystack.agent.step.llm` for the chat generator call, and one `haystack.agent.step.tool` span *per tool call* (only when tool calls happen). Previously each iteration produced two child spans through `Pipeline._run_component` (one for the chat generator, one for the tool invoker) tagged with `haystack.component.name` / `haystack.component.type`, and a single tool span grouped all of a step's tool calls together. The new spans do NOT carry `haystack.component.*` tags; the LLM span exposes content tags `haystack.agent.step.llm.input`/`.output`, and each tool span carries `haystack.tool.name` / `haystack.tool.description` tags plus content tags `haystack.agent.step.tool.input` (the call arguments) / `.output` (the tool result).

**Why:** Removes the dependency on `Pipeline._run_component` inside `Agent.run` and produces a clearer per-iteration trace structure that maps directly onto common agent-tracing conventions — one span per tool call, as codified by the OpenTelemetry GenAI "execute tool" span and followed by backends like Langfuse (`chain → {generation, tool, tool, …}`).

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
            # One span per tool call; the tool name rides along as a tag so the observation can be named upfront.
            tool_name = context.tags.get("haystack.tool.name")
            return LangfuseSpan(
                self.tracer.start_as_current_observation(
                    name=f"tool - {tool_name}" if tool_name else "tool",
                    as_type=cast(ObservationSpanType, "tool"),
                )
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

#### Prompt template variables are required by default

**What changed:** `Agent` now treats every Jinja2 template variable in `user_prompt` and `system_prompt` as required by default. The `required_variables` parameter's default has been changed from `None` (all optional) to `"*"` (all required). Previously, missing variables were silently rendered as empty strings. Passing `required_variables=None` explicitly still opts into the old "all optional" behavior.

**Why:** Avoids silent rendering bugs where a missing variable produces an unexpectedly empty section of the prompt. Aligns `Agent` with `LLM`, `PromptBuilder`, and `ChatPromptBuilder`, which already require all variables by default in v3.0.

**How to migrate:**

Before (v2.x):
```python
from haystack.components.agents import Agent

# All variables were optional by default; missing values rendered as "".
agent = Agent(
    chat_generator=...,
    tools=[...],
    user_prompt="Answer {{query}} in {{language}}.",
)
agent.run(messages=[], query="What is NLP?")  # language silently becomes ""
```

After (v3.0):
```python
from haystack.components.agents import Agent

# Option 1: provide every variable (matches the new safe default).
agent = Agent(
    chat_generator=...,
    tools=[...],
    user_prompt="Answer {{query}} in {{language}}.",
)
agent.run(messages=[], query="What is NLP?", language="English")

# Option 2: declare which variables are required; everything else stays optional.
agent = Agent(
    chat_generator=...,
    tools=[...],
    user_prompt="Answer {{query}} in {{language}}.",
    required_variables=["query"],
)
agent.run(messages=[], query="What is NLP?")  # language renders as ""

# Option 3: restore the old "all optional" behavior.
agent = Agent(
    chat_generator=...,
    tools=[...],
    user_prompt="Answer {{query}} in {{language}}.",
    required_variables=None,
)
agent.run(messages=[], query="What is NLP?")  # language renders as ""
```

#### Tools must declare `inputs_from_state` to read from `State` by name

**What changed:** A tool now reads a value from the Agent's `State` by name only when it declares an explicit `inputs_from_state` mapping. Previously, a tool without `inputs_from_state` had every parameter implicitly treated as a potential `State` key: any parameter whose name matched a `State` key (and that the LLM did not supply) was silently filled from `State`. This implicit name-matching has been removed. It applies to every tool type, since `ComponentTool`, `PipelineTool`, `MCPTool`, and others all derive from the base `Tool` class.

**Why:** The implicit matching was hidden — nothing in a tool's definition signaled that it read from `State`, so a tool could start consuming `State` purely because of an incidental parameter-name collision. With many tools and a large state schema this made accidental overlaps likely, and renaming a parameter or adding a `State` key could silently change what a tool received. Requiring an explicit mapping makes a tool's `State` dependencies visible and intentional. Auto-injection of the full `State` object into a parameter annotated as `State` is unaffected — that is explicit at the signature level.

**How to migrate:** Add an explicit `inputs_from_state` mapping (`{state_key: parameter_name}`) to any tool that should read from `State` by name.

Before (v2.x), whenever the model called `weather_tool` *without* supplying a `location` argument, the parameter was filled from the `location` state key automatically, because the parameter name matched a key in the Agent's `state_schema`. (If the model did supply `location`, the LLM-provided value always took precedence and `State` was not consulted.)
```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    function=weather_function,
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
    tools=[weather_tool],
    # `location` lives in State, so it was silently injected into the tool's same-named parameter
    # on any call where the model didn't already provide a `location` argument.
    state_schema={"location": {"type": str}},
)
```

After (v3.0), declare the mapping explicitly so the read from `State` is visible in the tool definition:
```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    function=weather_function,
    inputs_from_state={"location": "location"},
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
    tools=[weather_tool],
    state_schema={"location": {"type": str}},
)
```

Tools that take the full `State` object via a `State`-annotated parameter need no change:
```python
from typing import Annotated

from haystack.components.agents import State
from haystack.tools import Tool

def weather_function(location: Annotated[str, "The name of the city"], state: State) -> str:
    ...

weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    function=weather_function,  # `state` is still injected automatically
)
```

#### Reserved `state_schema` keys

**What changed:** `Agent` now reserves several names in its `state_schema`: the auto-populated run-metadata outputs `step_count`, `token_usage`, and `tool_call_counts`, and the hook-facing keys `continue_run` (set by an `on_exit` hook to keep the Agent running), `tools` (the tools available in the current step, for hooks to inspect), and `hook_context` (the per-run resources passed to `Agent.run`). Passing any of them as a `state_schema` key now raises `ValueError`.

**Why:** These keys are managed by `Agent` itself; allowing users to redefine them would let a user-defined entry shadow the value the Agent reads or writes.

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

#### Human-in-the-Loop confirmation is now a `before_tool` hook

**What changed:** The `confirmation_strategies` and `confirmation_strategy_context` parameters have been removed from `Agent.__init__`, `Agent.run`, and `Agent.run_async`. Human-in-the-Loop tool confirmation is now expressed through the general Agent hooks mechanism: wrap your confirmation strategies in a `ConfirmationHook` and register it under the `before_tool` hook point. Request-scoped resources that used to be passed via `confirmation_strategy_context` are now passed via the generic `hook_context` run argument (read by hooks with `state.get("hook_context")`).

**Why:** Confirmation was a one-off, before-tool interception bolted onto the Agent. Hooks generalize that seam, so HITL becomes one application of a single, uniform extension point instead of a parallel concept with its own serialization and run plumbing.

The Human-in-the-Loop module has also moved from `haystack.human_in_the_loop` to `haystack.hooks.human_in_the_loop`, so that it lives alongside the other built-in hooks (such as tool result offloading). Update your imports to the new location.

**How to migrate:**

Before (v2.x):
```python
from haystack.components.agents import Agent
from haystack.human_in_the_loop import BlockingConfirmationStrategy, AlwaysAskPolicy, RichConsoleUI

agent = Agent(
    chat_generator=...,
    tools=[...],
    confirmation_strategies={
        "my_tool": BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI()
        )
    },
)
agent.run(messages=[...], confirmation_strategy_context={"websocket": ws})
```

After (v3.0):
```python
from haystack.components.agents import Agent
from haystack.hooks.human_in_the_loop import (
    BlockingConfirmationStrategy,
    AlwaysAskPolicy,
    ConfirmationHook,
    RichConsoleUI,
)

confirmation_hook = ConfirmationHook(
    confirmation_strategies={
        "my_tool": BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI()
        )
    }
)
agent = Agent(chat_generator=..., tools=[...], hooks={"before_tool": [confirmation_hook]})
agent.run(messages=[...], hook_context={"websocket": ws})
```

#### Confirmation strategies now see only the model-requested tool arguments

**What changed:** Confirmation strategies now receive the arguments the model produced for a tool call in `tool_params`, rather than the fully-prepared arguments. Values injected from `State` via a tool's `inputs_from_state` mapping (and the `State` object passed to `State`-typed parameters) are no longer included in what is presented for confirmation — that injection now happens only at tool execution time.

**Why:** Preparing and baking each tool's arguments up front defeated the per-batch argument preparation in tool execution, so a tool that read a state key written by another tool in the same step could run with stale values. Confirmation now operates on the model-requested arguments and leaves state injection to execution. See the release note for the failure mode details.

**How to migrate:** If your `ConfirmationUI` or `ConfirmationPolicy` displayed or inspected state-injected argument values, update it to expect only the arguments the model provided. No change is needed if you only relied on the model-requested arguments.

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

### Pipeline deserialization is gated by a module allowlist

**What changed:** `Pipeline.load`, `Pipeline.loads`, and `Pipeline.from_dict` now refuse to import classes from modules outside a trusted-module allowlist and raise a `DeserializationError` instead. The default allowlist contains `haystack`, `haystack_integrations`, `haystack_experimental`, `builtins`, `typing`, and `collections`. Pipelines that reference custom components, callables, or types in other packages will fail to load until those modules are explicitly allowed.

In addition, `default_from_dict` now rejects nested `{"type": "..."}` dictionaries whose key is not an `__init__` parameter of the parent class — this can surface pre-existing YAML bugs (typos, leftovers from removed parameters, stale snapshots).

**Why:** Loading a pipeline from YAML used to dynamically import any class referenced in the file, which made a crafted YAML capable of causing arbitrary classes to be imported and instantiated. Gating imports through an allowlist closes that gap while leaving Haystack's own packages working out of the box.

**How to migrate:**

If your pipeline only references components from `haystack`, `haystack_integrations`, or `haystack_experimental`, no action is needed.

Otherwise, extend the allowlist via one of the four mechanisms below.

Before (v2.x), all modules implicitly trusted:
```python
from haystack import Pipeline

# Worked for any class on the import path, including third-party packages.
with open("pipeline.yaml") as fp:
    pipeline = Pipeline.load(fp)
```

After (v3.0), pick one of the following options. The first two scope the trust to a single call; the others extend it process-wide.

```python
# 1. Per-call kwarg — recommended for application code that knows exactly which extra
#    packages a given YAML needs.
from haystack import Pipeline

with open("pipeline.yaml") as fp:
    pipeline_a = Pipeline.load(fp, allowed_modules=["mypkg.*", "anotherpkg.components.*"])

# 2. Per-call bypass — equivalent to "I fully trust this YAML; skip the allowlist".
#    Mirrors the `yaml.safe_load` / `yaml.unsafe_load` convention.
with open("pipeline.yaml") as fp:
    pipeline_b = Pipeline.load(fp, unsafe=True)

# 3. Process-wide programmatic — call once at startup, e.g. in your application's
#    entry point or a custom integration package's __init__.
from haystack.core.serialization import allow_deserialization_module

allow_deserialization_module("mypkg.*")
with open("pipeline.yaml") as fp:
    pipeline_c = Pipeline.load(fp)  # `mypkg.*` is now trusted for every load in this process.
```

```bash
# 4. Environment variable — useful for ops/deployments where code shouldn't change.
#    Comma-separated patterns; read at runtime on every deserialization call.
export HAYSTACK_DESERIALIZATION_ALLOWLIST="mypkg.*,otherpkg.*"
```

Patterns are matched as prefixes by default (`"mypkg"` matches `mypkg` and any submodule), or as `fnmatch` globs if they contain `*`, `?`, or `[` somewhere other than a trailing `.*`.
### Generators removed

**What changed:** `OpenAIGenerator`, `AzureOpenAIGenerator`, `HuggingFaceAPIGenerator`, and `HuggingFaceLocalGenerator` have been removed.
Generators living in Haystack Core Integrations will also be removed soon.
Their chat counterparts are the replacement: `OpenAIChatGenerator` and `AzureOpenAIChatGenerator` in Haystack core, `HuggingFaceAPIChatGenerator` in the `huggingface-api-haystack` integration, and `TransformersChatGenerator` (the renamed `HuggingFaceLocalChatGenerator`) in the `transformers-haystack` integration (see [Components Moved to External Packages](#components-moved-to-external-packages)). As of Haystack 3.0, all ChatGenerators also accept a plain `str` as input, so the migration rarely requires structural changes.

**Why:** Over time, Generators became shallow wrappers over the ChatGenerators, converting `str → ChatMessage → str` around the exact same model calls. All new features (tool calling, structured outputs, etc.) were introduced only in ChatGenerators, leaving the legacy classes behind. They were also a source of confusion for newcomers and an unnecessary duplication of code and tests.

**How to migrate:**

#### Direct usage (running a generator from Python code)

Before (v2.x):
```python
from haystack.components.generators import OpenAIGenerator

gen = OpenAIGenerator()
result = gen.run("What is NLP?")
text = result["replies"][0]   # str
meta = result["meta"][0]      # dict with model metadata
```

After (v3.0):
```python
from haystack.components.generators.chat import OpenAIChatGenerator

gen = OpenAIChatGenerator()
result = gen.run("What is NLP?")   # str input accepted directly
reply = result["replies"][0]       # ChatMessage
text = reply.text                  # str
meta = reply.meta                  # dict with model metadata (now on the message)
```

#### System prompt

Before (v2.x):
```python
from haystack.components.generators import OpenAIGenerator

gen = OpenAIGenerator(system_prompt="You are concise.")
result = gen.run("What is NLP?")
```

After (v3.0):
```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

gen = OpenAIChatGenerator()
result = gen.run([
    ChatMessage.from_system("You are concise."),
    ChatMessage.from_user("What is NLP?"),
])
```

#### Pipeline usage

Pipelines that connected `PromptBuilder` (output: `str`) to a legacy Generator continue to work unchanged when you swap in a ChatGenerator. The Haystack pipeline type system automatically converts `str` to `list[ChatMessage]` at the connection edge.

Before (v2.x):
```python
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("llm", OpenAIGenerator())
pipeline.connect("prompt_builder", "llm")   # str → str
```

After (v3.0), minimal change (smart connection still works):
```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import PromptBuilder

pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component("llm", OpenAIChatGenerator())
pipeline.connect("prompt_builder", "llm")   # str → list[ChatMessage], auto-converted
```

Alternatively, for an idiomatic v3 pipeline use `ChatPromptBuilder`:
```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [ChatMessage.from_user(prompt_template)]
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
pipeline.add_component("llm", OpenAIChatGenerator())
pipeline.connect("prompt_builder.prompt", "llm.messages")
```

#### `meta` output socket removed

Legacy Generators exposed a second output socket `meta: list[dict]`. ChatGenerators do not; per-reply metadata is embedded in each `ChatMessage.meta`. If you had a pipeline connection to `llm.meta`, remove it. `AnswerBuilder` already handles this automatically (it reads metadata from `ChatMessage.meta` when the replies are `list[ChatMessage]`).

Before (v2.x):
```python
pipeline.connect("llm.replies", "answer_builder.replies")
pipeline.connect("llm.meta", "answer_builder.meta")   # separate meta socket
```

After (v3.0):
```python
pipeline.connect("llm.replies", "answer_builder.replies")
# no meta connection needed; AnswerBuilder reads it from ChatMessage.meta
```

### `DALLEImageGenerator` renamed to `OpenAIImageGenerator`

**What changed:** `DALLEImageGenerator` has been renamed to `OpenAIImageGenerator` and moved to `haystack.components.generators.openai_image_generator`. The API and parameters are otherwise unchanged.

**Why:** OpenAI retired the DALL-E model family. The new name reflects that the component works with the full OpenAI image generation API and is no longer tied to a specific model family.

**How to migrate:**

Before (v2.x):
```python
from haystack.components.generators import DALLEImageGenerator

generator = DALLEImageGenerator(model="dall-e-3")
result = generator.run("A photo of a red apple")
```

After (v3.0):
```python
from haystack.components.generators import OpenAIImageGenerator

generator = OpenAIImageGenerator(model="gpt-image-2")
result = generator.run("A photo of a red apple")
```

### `AsyncPipeline` merged into `Pipeline`

**What changed:** The `AsyncPipeline` class has been removed. Its asynchronous methods (`run_async`, `run_async_generator`, `stream`) are now part of the single `Pipeline` class, alongside the synchronous `run`.

**Why:** Two classes caused friction where sync and async met: `AsyncPipeline.run()` wrapped `asyncio.run()` and raised inside an already-running event loop (e.g. Jupyter, FastAPI), and a `SuperComponent` exposed `run_async` even for sync pipelines, where it always failed. A single `Pipeline` with native `run` and `run_async` fixes both.

**How to migrate:**

Replace `AsyncPipeline` with `Pipeline`; the async methods are unchanged.

Before (v2.x):
```python
from haystack import AsyncPipeline

pipeline = AsyncPipeline()
result = await pipeline.run_async(data)
```

After (v3.0):
```python
from haystack import Pipeline

pipeline = Pipeline()
result = await pipeline.run_async(data)
```

If you used the **synchronous** `AsyncPipeline.run()`, note it was a wrapper around the concurrent async engine, so `Pipeline.run()` is not a drop-in replacement. Choose by intent:

```python
# Keep concurrent execution from sync code:
result = asyncio.run(pipeline.run_async(data, concurrency_limit=4))

# Sequential execution is fine:
result = pipeline.run(data)  # components run one at a time; no concurrency_limit
```

Unlike `AsyncPipeline.run()`, `Pipeline.run()` does not raise when called inside a running event loop: it runs and blocks the loop. In an async context, use `await pipeline.run_async(...)`.

**Behavior to be aware of:**

- `Pipeline.run` runs components sequentially and does not accept `concurrency_limit`; only `run_async` / `run_async_generator` run components concurrently.
- Only `run` supports breakpoints (`break_point` / `pipeline_snapshot`).
- Both run paths are traced under a single `haystack.pipeline.run` operation name, distinguished by a `haystack.pipeline.execution_mode` tag (`sync` or `async`); previously asynchronous runs used `haystack.async_pipeline.run`.

### Auto-generated `Document.id` changes for documents with non-empty `meta`

**What changed:** The hash used to auto-generate `Document.id` is now computed from a canonical (key-sorted) JSON serialization of `meta` instead of the dict's `repr`. Documents with empty `meta` keep the same IDs as before, but documents with non-empty `meta` get different IDs in v3.0. Non-JSON-serializable `meta` values (e.g. `datetime` or custom classes) are now serialized via `str(...)` rather than `repr(...)`, which also changes their IDs. See [#11446](https://github.com/deepset-ai/haystack/pull/11446).

**Why:** Previously the hash reflected the insertion order of keys in `meta`, so two documents with the same content and the same metadata could end up with different IDs depending on how the `meta` dict was constructed. This silently broke `DuplicatePolicy.SKIP` / `FAIL` and any cache or dedup table keyed on the document ID. Sorting the keys before hashing makes the ID order-independent.

**How to migrate:**

If you rely on auto-generated IDs to match documents already persisted in a `DocumentStore` written by Haystack v2.x, re-ingest the affected documents so the new IDs are used consistently, or pass the previous `id` explicitly when constructing the `Document`.

Before (v2.x):
```python
from haystack.dataclasses import Document

# ID was derived from meta's dict repr, so it depended on key insertion order:
# these two documents could end up with different IDs.
doc1 = Document(content="Berlin is the capital of Germany.", meta={"source": "wiki", "lang": "en"})
doc2 = Document(content="Berlin is the capital of Germany.", meta={"lang": "en", "source": "wiki"})

After (v3.0):
```python
from haystack.dataclasses import Document

# Same content + meta now always yields the same ID, regardless of key order,
# but that ID differs from the one v2.x produced for documents with non-empty meta.
doc1 = Document(content="Berlin is the capital of Germany.", meta={"source": "wiki", "lang": "en"})
doc2 = Document(content="Berlin is the capital of Germany.", meta={"lang": "en", "source": "wiki"})
assert doc1.id == doc2.id
```

It is possible to migrate an existing index without rerunning your indexing pipeline, for example to avoid recalculating embeddings. To do that, read stored documents, regenerate their IDs using Haystack 3.0, write the updated documents, and delete the documents stored under their old IDs.

```python
from dataclasses import replace

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

# Example DocumentStore with IDs generated with Haystack 2.x
store = InMemoryDocumentStore()
store.write_documents(
    [
        Document(
            id="b51c3ee6b892f52bf28af01f5d823a254e438356ec335a20133ad940ef7b8cd7",
            content="Berlin is the capital of Germany.",
            meta={"source": "wiki", "lang": "en"},
        ),
        Document(
            id="f022d8d89a99f89547215f8adcfed92f41518f2bb3e11d14e27987bd9d265ead",
            content="Paris is the capital of France.",
            meta={"source": "wiki", "lang": "en"},
        ),
    ],
    policy=DuplicatePolicy.OVERWRITE,
)

# Exemplary steps to re-calculate IDs. Note that all documents are retrieved at once in this example but larger indices require pagination.
old_documents = store.filter_documents()
new_documents = [replace(doc, id="") for doc in old_documents]
store.write_documents(new_documents, policy=DuplicatePolicy.OVERWRITE)
new_ids = {doc.id for doc in new_documents}
store.delete_documents([doc.id for doc in old_documents if doc.id not in new_ids])
```

### Haystack logging no longer reconfigures logging for the whole process

**What changed:** Importing Haystack no longer attaches its formatting handler to the root logger, and no longer
configures `structlog` process-wide. The handler is now scoped to Haystack's own logger namespaces (`haystack`,
`haystack_integrations`, `haystack_experimental`), and the global `structlog` configuration is set only when you call
`configure_logging()` explicitly. As a result, importing Haystack no longer reformats the logs of the host application
or other libraries running in the same process.

**Why:** Haystack should behave as a well-mannered library when it runs alongside other services in the same process,
rather than taking over logging for the whole process.

**How to migrate:** If you relied on Haystack formatting every log record in the process, opt back in explicitly.

Before (v2.x):
```python
import haystack  # formatted every log record in the process and configured structlog globally
```

After (v3.0):
```python
from haystack import logging

# Restore the old behavior: format every log record in the process (also configures structlog globally).
logging.configure_logging(logger_name="")
```

**Note on duplicate log lines:** Haystack's handler now sits on the `haystack.*` loggers, which still propagate to the
root logger. If the host application also configures a handler on the root logger, Haystack's own logs can appear
twice. To make Haystack fully own its output and stop the duplication, disable propagation:

```python
from haystack import logging

logging.configure_logging(propagate=False)
```

### Components now resolve API keys at warm-up

**What changed:** Components that use external services now create their resources (such as API clients) during `warm_up()` instead of in `__init__`. As a consequence, a missing API key (for example, an unset environment variable behind a `Secret.from_env_var` default) is now reported at warm-up or first run rather than at construction. This affects OpenAI and Azure OpenAI components.

**Why:** Creating resources in `warm_up()` / `warm_up_async()` and releasing them in `close()` / `close_async()` gives components and pipelines a single, predictable resource lifecycle.

**How to migrate:** If you relied on construction failing for a missing API key, expect the same error at `warm_up()` (or the first `run`) instead.

Before (v2.x), with `OPENAI_API_KEY` unset:
```python
from haystack.components.embedders import OpenAITextEmbedder

embedder = OpenAITextEmbedder()  # raised here
```

After (v3.0), with `OPENAI_API_KEY` unset:
```python
from haystack.components.embedders import OpenAITextEmbedder

embedder = OpenAITextEmbedder()  # no error at construction
embedder.warm_up()               # raised here
```
