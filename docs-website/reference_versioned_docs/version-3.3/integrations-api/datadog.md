---
title: "Datadog"
id: integrations-datadog
description: "Datadog integration for Haystack"
slug: "/integrations-datadog"
---


## haystack_integrations.components.connectors.datadog.datadog_connector

### DatadogConnector

DatadogConnector connects Haystack to [Datadog](https://www.datadoghq.com/) in order to enable the tracing of

operations and data flow within the components of a pipeline.

To use the DatadogConnector, add it to your pipeline without connecting it to any other component. It will
automatically trace all pipeline operations when tracing is enabled.

**Environment Configuration:**

- `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be set to `"true"` to trace the content (inputs and outputs) of the
  pipeline components.
- Datadog is configured through the standard `ddtrace` mechanisms, e.g. the `DD_SERVICE`, `DD_ENV` and
  `DD_VERSION` environment variables or by running your application with the `ddtrace-run` command. See the
  [ddtrace documentation](https://ddtrace.readthedocs.io/en/stable/) for more details.

Here is an example of how to use the DatadogConnector in a pipeline:

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.connectors.datadog import DatadogConnector

pipe = Pipeline()
pipe.add_component("tracer", DatadogConnector("Chat example"))
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
__init__(name: str = 'datadog') -> None
```

Initialize the DatadogConnector component.

**Parameters:**

- **name** (<code>str</code>) – The name used to identify this tracing component. It is returned by the `run` method and can be
  used to mark traces produced by this connector.

#### run

```python
run() -> dict[str, str]
```

Runs the DatadogConnector component.

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
from_dict(data: dict[str, Any]) -> DatadogConnector
```

Deserialize this component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – The dictionary representation of this component.

**Returns:**

- <code>DatadogConnector</code> – The deserialized component instance.

## haystack_integrations.tracing.datadog.tracer

### DatadogSpan

Bases: <code>Span</code>

#### __init__

```python
__init__(span: ddSpan) -> None
```

Creates an instance of DatadogSpan.

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

### DatadogTracer

Bases: <code>Tracer</code>

#### __init__

```python
__init__(tracer: ddTracer) -> None
```

Creates an instance of DatadogTracer.

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
