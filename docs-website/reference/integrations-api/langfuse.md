---
title: "langfuse"
id: integrations-langfuse
description: "Langfuse integration for Haystack"
slug: "/integrations-langfuse"
---


## `haystack_integrations.components.connectors.langfuse.langfuse_connector`

### `LangfuseConnector`

LangfuseConnector connects Haystack LLM framework with [Langfuse](https://langfuse.com) in order to enable the
tracing of operations and data flow within various components of a pipeline.

To use LangfuseConnector, add it to your pipeline without connecting it to any other components.
It will automatically trace all pipeline operations when tracing is enabled.

**Environment Configuration:**

- `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY`: Required Langfuse API credentials.
- `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to enable tracing.
- `HAYSTACK_LANGFUSE_ENFORCE_FLUSH`: (Optional) If set to `"false"`, disables flushing after each component.
  Be cautious: this may cause data loss on crashes unless you manually flush before shutdown.
  By default, the data is flushed after each component and blocks the thread until the data is sent to Langfuse.

If you disable flushing after each component make sure you will call langfuse.flush() explicitly before the
program exits. For example:

```python
from haystack.tracing import tracer

try:
    # your code here
finally:
    tracer.actual_tracer.flush()
```

or in FastAPI by defining a shutdown event handler:

```python
from haystack.tracing import tracer

# ...

@app.on_event("shutdown")
async def shutdown_event():
    tracer.actual_tracer.flush()
```

Here is an example of how to use LangfuseConnector in a pipeline:

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.langfuse import (
    LangfuseConnector,
)

pipe = Pipeline()
pipe.add_component("tracer", LangfuseConnector("Chat example"))
pipe.add_component("prompt_builder", ChatPromptBuilder())
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

pipe.connect("prompt_builder.prompt", "llm.messages")

messages = [
    ChatMessage.from_system(
        "Always respond in German even if some input data is in other languages."
    ),
    ChatMessage.from_user("Tell me about {{location}}"),
]

response = pipe.run(
    data={
        "prompt_builder": {
            "template_variables": {"location": "Berlin"},
            "template": messages,
        }
    }
)
print(response["llm"]["replies"][0])
print(response["tracer"]["trace_url"])
print(response["tracer"]["trace_id"])
```

For advanced use cases, you can also customize how spans are created and processed by providing a custom
SpanHandler. This allows you to add custom metrics, set warning levels, or attach additional metadata to your
Langfuse traces:

```python
from haystack_integrations.tracing.langfuse import DefaultSpanHandler, LangfuseSpan
from typing import Optional

class CustomSpanHandler(DefaultSpanHandler):

    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        # Custom span handling logic, customize Langfuse spans however it fits you
        # see DefaultSpanHandler for how we create and process spans by default
        pass

connector = LangfuseConnector(span_handler=CustomSpanHandler())
```

#### `__init__`

```python
__init__(
    name: str,
    public: bool = False,
    public_key: Secret | None = Secret.from_env_var("LANGFUSE_PUBLIC_KEY"),
    secret_key: Secret | None = Secret.from_env_var("LANGFUSE_SECRET_KEY"),
    httpx_client: httpx.Client | None = None,
    span_handler: SpanHandler | None = None,
    *,
    host: str | None = None,
    langfuse_client_kwargs: dict[str, Any] | None = None
) -> None
```

Initialize the LangfuseConnector component.

**Parameters:**

- **name** (<code>str</code>) – The name for the trace. This name will be used to identify the tracing run in the Langfuse
  dashboard.
- **public** (<code>bool</code>) – Whether the tracing data should be public or private. If set to `True`, the tracing data will be
  publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private and
  only accessible to the Langfuse account owner. The default is `False`.
- **public_key** (<code>Secret | None</code>) – The Langfuse public key. Defaults to reading from LANGFUSE_PUBLIC_KEY environment variable.
- **secret_key** (<code>Secret | None</code>) – The Langfuse secret key. Defaults to reading from LANGFUSE_SECRET_KEY environment variable.
- **httpx_client** (<code>Client | None</code>) – Optional custom httpx.Client instance to use for Langfuse API calls. Note that when
  deserializing a pipeline from YAML, any custom client is discarded and Langfuse will create its own default
  client, since HTTPX clients cannot be serialized.
- **span_handler** (<code>SpanHandler | None</code>) – Optional custom handler for processing spans. If None, uses DefaultSpanHandler.
  The span handler controls how spans are created and processed, allowing customization of span types
  based on component types and additional processing after spans are yielded. See SpanHandler class for
  details on implementing custom handlers.
  host: Host of Langfuse API. Can also be set via `LANGFUSE_HOST` environment variable.
  By default it is set to `https://cloud.langfuse.com`.
- **langfuse_client_kwargs** (<code>dict\[str, Any\] | None</code>) – Optional custom configuration for the Langfuse client. This is a dictionary
  containing any additional configuration options for the Langfuse client. See the Langfuse documentation
  for more details on available configuration options.

#### `run`

```python
run(invocation_context: dict[str, Any] | None = None) -> dict[str, str]
```

Runs the LangfuseConnector component.

**Parameters:**

- **invocation_context** (<code>dict\[str, Any\] | None</code>) – A dictionary with additional context for the invocation. This parameter
  is useful when users want to mark this particular invocation with additional information, e.g.
  a run id from their own execution framework, user id, etc. These key-value pairs are then visible
  in the Langfuse traces.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with the following keys:
- `name`: The name of the tracing component.
- `trace_url`: The URL to the tracing data.
- `trace_id`: The ID of the trace.

#### `to_dict`

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### `from_dict`

```python
from_dict(data: dict[str, Any]) -> LangfuseConnector
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>LangfuseConnector</code> – The deserialized component instance.

## `haystack_integrations.tracing.langfuse.tracer`

### `LangfuseSpan`

Bases: <code>Span</code>

Internal class representing a bridge between the Haystack span tracing API and Langfuse.

#### `__init__`

```python
__init__(context_manager: AbstractContextManager) -> None
```

Initialize a LangfuseSpan instance.

**Parameters:**

- **context_manager** (<code>AbstractContextManager</code>) – The context manager from Langfuse created with
  `langfuse.get_client().start_as_current_span` or
  `langfuse.get_client().start_as_current_observation`.

#### `set_tag`

```python
set_tag(key: str, value: Any) -> None
```

Set a generic tag for this span.

**Parameters:**

- **key** (<code>str</code>) – The tag key.
- **value** (<code>Any</code>) – The tag value.

#### `set_content_tag`

```python
set_content_tag(key: str, value: Any) -> None
```

Set a content-specific tag for this span.

**Parameters:**

- **key** (<code>str</code>) – The content tag key.
- **value** (<code>Any</code>) – The content tag value.

#### `raw_span`

```python
raw_span() -> LangfuseClientSpan
```

Return the underlying span instance.

**Returns:**

- <code>LangfuseSpan</code> – The Langfuse span instance.

#### `get_data`

```python
get_data() -> dict[str, Any]
```

Return the data associated with the span.

**Returns:**

- <code>dict\[str, Any\]</code> – The data associated with the span.

### `SpanContext`

Context for creating spans in Langfuse.

Encapsulates the information needed to create and configure a span in Langfuse tracing.
Used by SpanHandler to determine the span type (trace, generation, or default) and its configuration.

**Parameters:**

- **name** (<code>str</code>) – The name of the span to create. For components, this is typically the component name.
- **operation_name** (<code>str</code>) – The operation being traced (e.g. "haystack.pipeline.run"). Used to determine
  if a new trace should be created without warning.
- **component_type** (<code>str | None</code>) – The type of component creating the span (e.g. "OpenAIChatGenerator").
  Can be used to determine the type of span to create.
- **tags** (<code>dict\[str, Any\]</code>) – Additional metadata to attach to the span. Contains component input/output data
  and other trace information.
- **parent_span** (<code>Span | None</code>) – The parent span if this is a child span. If None, a new trace will be created.
- **trace_name** (<code>str</code>) – The name to use for the trace when creating a parent span. Defaults to "Haystack".
- **public** (<code>bool</code>) – Whether traces should be publicly accessible. Defaults to False.

### `SpanHandler`

Bases: <code>ABC</code>

Abstract base class for customizing how Langfuse spans are created and processed.

This class defines two key extension points:

1. create_span: Controls what type of span to create (default or generation)
1. handle: Processes the span after component execution (adding metadata, metrics, etc.)

To implement a custom handler:

- Extend this class or DefaultSpanHandler
- Override create_span and handle methods. It is more common to override handle.
- Pass your handler to LangfuseConnector init method

#### `init_tracer`

```python
init_tracer(tracer: langfuse.Langfuse) -> None
```

Initialize with Langfuse tracer. Called internally by LangfuseTracer.

**Parameters:**

- **tracer** (<code>Langfuse</code>) – The Langfuse client instance to use for creating spans

#### `create_span`

```python
create_span(context: SpanContext) -> LangfuseSpan
```

Create a span of appropriate type based on the context.

This method determines what kind of span to create:

- A new trace if there's no parent span
- A generation span for LLM components
- A default span for other components

**Parameters:**

- **context** (<code>SpanContext</code>) – The context containing all information needed to create the span

**Returns:**

- <code>LangfuseSpan</code> – A new LangfuseSpan instance configured according to the context

#### `handle`

```python
handle(span: LangfuseSpan, component_type: str | None) -> None
```

Process a span after component execution by attaching metadata and metrics.

This method is called after the component or pipeline yields its span, allowing you to:

- Extract and attach token usage statistics
- Add model information
- Record timing data (e.g., time-to-first-token)
- Set log levels for quality monitoring
- Add custom metrics and observations

**Parameters:**

- **span** (<code>LangfuseSpan</code>) – The span that was yielded by the component
- **component_type** (<code>str | None</code>) – The type of component that created the span, used to determine
  what metadata to extract and how to process it

### `DefaultSpanHandler`

Bases: <code>SpanHandler</code>

DefaultSpanHandler provides the default Langfuse tracing behavior for Haystack.

### `LangfuseTracer`

Bases: <code>Tracer</code>

Internal class representing a bridge between the Haystack tracer and Langfuse.

#### `__init__`

```python
__init__(
    tracer: langfuse.Langfuse,
    name: str = "Haystack",
    public: bool = False,
    span_handler: SpanHandler | None = None,
) -> None
```

Initialize a LangfuseTracer instance.

**Parameters:**

- **tracer** (<code>Langfuse</code>) – The Langfuse tracer instance.
- **name** (<code>str</code>) – The name of the pipeline or component. This name will be used to identify the tracing run on the
  Langfuse dashboard.
- **public** (<code>bool</code>) – Whether the tracing data should be public or private. If set to `True`, the tracing data will
  be publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private
  and only accessible to the Langfuse account owner.
- **span_handler** (<code>SpanHandler | None</code>) – Custom handler for processing spans. If None, uses DefaultSpanHandler.

#### `current_span`

```python
current_span() -> Span | None
```

Return the current active span.

**Returns:**

- <code>Span | None</code> – The current span if available, else None.

#### `get_trace_url`

```python
get_trace_url() -> str
```

Return the URL to the tracing data.

**Returns:**

- <code>str</code> – The URL to the tracing data.

#### `get_trace_id`

```python
get_trace_id() -> str
```

Return the trace ID.

**Returns:**

- <code>str</code> – The trace ID.
