---
title: "langfuse"
id: integrations-langfuse
description: "Langfuse integration for Haystack"
slug: "/integrations-langfuse"
---

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector"></a>

## Module haystack\_integrations.components.connectors.langfuse.langfuse\_connector

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector"></a>

### LangfuseConnector

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

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector.__init__"></a>

#### LangfuseConnector.\_\_init\_\_

```python
def __init__(name: str,
             public: bool = False,
             public_key: Optional[Secret] = Secret.from_env_var(
                 "LANGFUSE_PUBLIC_KEY"),
             secret_key: Optional[Secret] = Secret.from_env_var(
                 "LANGFUSE_SECRET_KEY"),
             httpx_client: Optional[httpx.Client] = None,
             span_handler: Optional[SpanHandler] = None,
             *,
             host: Optional[str] = None,
             langfuse_client_kwargs: Optional[Dict[str, Any]] = None) -> None
```

Initialize the LangfuseConnector component.

**Arguments**:

- `name`: The name for the trace. This name will be used to identify the tracing run in the Langfuse
dashboard.
- `public`: Whether the tracing data should be public or private. If set to `True`, the tracing data will be
publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private and
only accessible to the Langfuse account owner. The default is `False`.
- `public_key`: The Langfuse public key. Defaults to reading from LANGFUSE_PUBLIC_KEY environment variable.
- `secret_key`: The Langfuse secret key. Defaults to reading from LANGFUSE_SECRET_KEY environment variable.
- `httpx_client`: Optional custom httpx.Client instance to use for Langfuse API calls. Note that when
deserializing a pipeline from YAML, any custom client is discarded and Langfuse will create its own default
client, since HTTPX clients cannot be serialized.
- `span_handler`: Optional custom handler for processing spans. If None, uses DefaultSpanHandler.
The span handler controls how spans are created and processed, allowing customization of span types
    based on component types and additional processing after spans are yielded. See SpanHandler class for
    details on implementing custom handlers.
host: Host of Langfuse API. Can also be set via `LANGFUSE_HOST` environment variable.
    By default it is set to `https://cloud.langfuse.com`.
- `langfuse_client_kwargs`: Optional custom configuration for the Langfuse client. This is a dictionary
containing any additional configuration options for the Langfuse client. See the Langfuse documentation
for more details on available configuration options.

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector.run"></a>

#### LangfuseConnector.run

```python
@component.output_types(name=str, trace_url=str, trace_id=str)
def run(invocation_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]
```

Runs the LangfuseConnector component.

**Arguments**:

- `invocation_context`: A dictionary with additional context for the invocation. This parameter
is useful when users want to mark this particular invocation with additional information, e.g.
a run id from their own execution framework, user id, etc. These key-value pairs are then visible
in the Langfuse traces.

**Returns**:

A dictionary with the following keys:
- `name`: The name of the tracing component.
- `trace_url`: The URL to the tracing data.
- `trace_id`: The ID of the trace.

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector.to_dict"></a>

#### LangfuseConnector.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="haystack_integrations.components.connectors.langfuse.langfuse_connector.LangfuseConnector.from_dict"></a>

#### LangfuseConnector.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "LangfuseConnector"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="haystack_integrations.tracing.langfuse.tracer"></a>

## Module haystack\_integrations.tracing.langfuse.tracer

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan"></a>

### LangfuseSpan

Internal class representing a bridge between the Haystack span tracing API and Langfuse.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan.__init__"></a>

#### LangfuseSpan.\_\_init\_\_

```python
def __init__(context_manager: AbstractContextManager) -> None
```

Initialize a LangfuseSpan instance.

**Arguments**:

- `context_manager`: The context manager from Langfuse created with
`langfuse.get_client().start_as_current_span` or
`langfuse.get_client().start_as_current_observation`.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan.set_tag"></a>

#### LangfuseSpan.set\_tag

```python
def set_tag(key: str, value: Any) -> None
```

Set a generic tag for this span.

**Arguments**:

- `key`: The tag key.
- `value`: The tag value.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan.set_content_tag"></a>

#### LangfuseSpan.set\_content\_tag

```python
def set_content_tag(key: str, value: Any) -> None
```

Set a content-specific tag for this span.

**Arguments**:

- `key`: The content tag key.
- `value`: The content tag value.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan.raw_span"></a>

#### LangfuseSpan.raw\_span

```python
def raw_span() -> LangfuseClientSpan
```

Return the underlying span instance.

**Returns**:

The Langfuse span instance.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseSpan.get_data"></a>

#### LangfuseSpan.get\_data

```python
def get_data() -> Dict[str, Any]
```

Return the data associated with the span.

**Returns**:

The data associated with the span.

<a id="haystack_integrations.tracing.langfuse.tracer.SpanContext"></a>

### SpanContext

Context for creating spans in Langfuse.

Encapsulates the information needed to create and configure a span in Langfuse tracing.
Used by SpanHandler to determine the span type (trace, generation, or default) and its configuration.

**Arguments**:

- `name`: The name of the span to create. For components, this is typically the component name.
- `operation_name`: The operation being traced (e.g. "haystack.pipeline.run"). Used to determine
if a new trace should be created without warning.
- `component_type`: The type of component creating the span (e.g. "OpenAIChatGenerator").
Can be used to determine the type of span to create.
- `tags`: Additional metadata to attach to the span. Contains component input/output data
and other trace information.
- `parent_span`: The parent span if this is a child span. If None, a new trace will be created.
- `trace_name`: The name to use for the trace when creating a parent span. Defaults to "Haystack".
- `public`: Whether traces should be publicly accessible. Defaults to False.

<a id="haystack_integrations.tracing.langfuse.tracer.SpanContext.__post_init__"></a>

#### SpanContext.\_\_post\_init\_\_

```python
def __post_init__() -> None
```

Validate the span context attributes.

**Raises**:

- `ValueError`: If name, operation_name or trace_name are empty
- `TypeError`: If tags is not a dictionary

<a id="haystack_integrations.tracing.langfuse.tracer.SpanHandler"></a>

### SpanHandler

Abstract base class for customizing how Langfuse spans are created and processed.

This class defines two key extension points:
1. create_span: Controls what type of span to create (default or generation)
2. handle: Processes the span after component execution (adding metadata, metrics, etc.)

To implement a custom handler:
- Extend this class or DefaultSpanHandler
- Override create_span and handle methods. It is more common to override handle.
- Pass your handler to LangfuseConnector init method

<a id="haystack_integrations.tracing.langfuse.tracer.SpanHandler.init_tracer"></a>

#### SpanHandler.init\_tracer

```python
def init_tracer(tracer: langfuse.Langfuse) -> None
```

Initialize with Langfuse tracer. Called internally by LangfuseTracer.

**Arguments**:

- `tracer`: The Langfuse client instance to use for creating spans

<a id="haystack_integrations.tracing.langfuse.tracer.SpanHandler.create_span"></a>

#### SpanHandler.create\_span

```python
@abstractmethod
def create_span(context: SpanContext) -> LangfuseSpan
```

Create a span of appropriate type based on the context.

This method determines what kind of span to create:
- A new trace if there's no parent span
- A generation span for LLM components
- A default span for other components

**Arguments**:

- `context`: The context containing all information needed to create the span

**Returns**:

A new LangfuseSpan instance configured according to the context

<a id="haystack_integrations.tracing.langfuse.tracer.SpanHandler.handle"></a>

#### SpanHandler.handle

```python
@abstractmethod
def handle(span: LangfuseSpan, component_type: Optional[str]) -> None
```

Process a span after component execution by attaching metadata and metrics.

This method is called after the component or pipeline yields its span, allowing you to:
- Extract and attach token usage statistics
- Add model information
- Record timing data (e.g., time-to-first-token)
- Set log levels for quality monitoring
- Add custom metrics and observations

**Arguments**:

- `span`: The span that was yielded by the component
- `component_type`: The type of component that created the span, used to determine
what metadata to extract and how to process it

<a id="haystack_integrations.tracing.langfuse.tracer.DefaultSpanHandler"></a>

### DefaultSpanHandler

DefaultSpanHandler provides the default Langfuse tracing behavior for Haystack.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseTracer"></a>

### LangfuseTracer

Internal class representing a bridge between the Haystack tracer and Langfuse.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseTracer.__init__"></a>

#### LangfuseTracer.\_\_init\_\_

```python
def __init__(tracer: langfuse.Langfuse,
             name: str = "Haystack",
             public: bool = False,
             span_handler: Optional[SpanHandler] = None) -> None
```

Initialize a LangfuseTracer instance.

**Arguments**:

- `tracer`: The Langfuse tracer instance.
- `name`: The name of the pipeline or component. This name will be used to identify the tracing run on the
Langfuse dashboard.
- `public`: Whether the tracing data should be public or private. If set to `True`, the tracing data will
be publicly accessible to anyone with the tracing URL. If set to `False`, the tracing data will be private
and only accessible to the Langfuse account owner.
- `span_handler`: Custom handler for processing spans. If None, uses DefaultSpanHandler.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseTracer.current_span"></a>

#### LangfuseTracer.current\_span

```python
def current_span() -> Optional[Span]
```

Return the current active span.

**Returns**:

The current span if available, else None.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseTracer.get_trace_url"></a>

#### LangfuseTracer.get\_trace\_url

```python
def get_trace_url() -> str
```

Return the URL to the tracing data.

**Returns**:

The URL to the tracing data.

<a id="haystack_integrations.tracing.langfuse.tracer.LangfuseTracer.get_trace_id"></a>

#### LangfuseTracer.get\_trace\_id

```python
def get_trace_id() -> str
```

Return the trace ID.

**Returns**:

The trace ID.

