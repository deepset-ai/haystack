---
title: "OpenTelemetry"
id: integrations-opentelemetry
description: "OpenTelemetry integration for Haystack"
slug: "/integrations-opentelemetry"
---


## haystack_integrations.components.connectors.opentelemetry.opentelemetry_connector

### OpenTelemetryConnector

OpenTelemetryConnector connects Haystack to [OpenTelemetry](https://opentelemetry.io/) in order to enable the

tracing of operations and data flow within the components of a pipeline.

To use the OpenTelemetryConnector, add it to your pipeline without connecting it to any other component. It will
automatically trace all pipeline operations when tracing is enabled. Make sure to configure an OpenTelemetry
`TracerProvider` (for example, with an exporter) before initializing the connector.

**Environment Configuration:**

- `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to trace the content (inputs and outputs) of the
  pipeline components.

Here is an example of how to use the OpenTelemetryConnector in a pipeline:

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

# Configure the OpenTelemetry SDK. A service name is required for most backends.
resource = Resource(attributes={ResourceAttributes.SERVICE_NAME: "haystack"})
tracer_provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
tracer_provider.add_span_processor(processor)
trace.set_tracer_provider(tracer_provider)

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.opentelemetry import OpenTelemetryConnector

pipe = Pipeline()
pipe.add_component("tracer", OpenTelemetryConnector())
pipe.add_component("prompt_builder", ChatPromptBuilder())
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

pipe.connect("prompt_builder.prompt", "llm.messages")

messages = [
    ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
    ChatMessage.from_user("Tell me about {{location}}"),
]

response = pipe.run(
    data={"prompt_builder": {"template_variables": {"location": "Berlin"}, "template": messages}}
)
print(response["llm"]["replies"][0])
```

#### __init__

```python
__init__(name: str = 'opentelemetry') -> None
```

Initialize the OpenTelemetryConnector component.

**Parameters:**

- **name** (<code>str</code>) – The name used to identify this tracing component. It is returned by the `run` method and can be
  used to mark traces produced by this connector.

#### run

```python
run() -> dict[str, str]
```

Runs the OpenTelemetryConnector component.

**Returns:**

- <code>dict\[str, str\]</code> – A dictionary with the following keys:
- `name`: The name of the tracing component.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – The serialized component as a dictionary.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> OpenTelemetryConnector
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>OpenTelemetryConnector</code> – The deserialized component instance.

## haystack_integrations.tracing.opentelemetry.tracer

### OpenTelemetrySpan

Bases: <code>Span</code>

#### __init__

```python
__init__(span: opentelemetry.trace.Span) -> None
```

Creates an instance of OpenTelemetrySpan.

#### set_tag

```python
set_tag(key: str, value: Any) -> None
```

Set a single tag on the span.

**Parameters:**

- **key** (<code>str</code>) – the name of the tag.
- **value** (<code>Any</code>) – the value of the tag.

#### raw_span

```python
raw_span() -> Any
```

Provides access to the underlying span object of the tracer.

**Returns:**

- <code>Any</code> – The underlying span object.

#### get_correlation_data_for_logs

```python
get_correlation_data_for_logs() -> dict[str, Any]
```

Return a dictionary with correlation data for logs.

### OpenTelemetryTracer

Bases: <code>Tracer</code>

#### __init__

```python
__init__(tracer: opentelemetry.trace.Tracer) -> None
```

Creates an instance of OpenTelemetryTracer.

#### trace

```python
trace(
    operation_name: str,
    tags: dict[str, Any] | None = None,
    parent_span: Span | None = None,
) -> Iterator[Span]
```

Activate and return a new span that inherits from the current active span.

#### current_span

```python
current_span() -> Span | None
```

Return the current active span
