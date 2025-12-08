---
title: "weights and bias"
id: integrations-weights-bias
description: "Weights & Bias integration for Haystack"
slug: "/integrations-weights-bias"
---

<a id="haystack_integrations.components.connectors.weave.weave_connector"></a>

## Module haystack\_integrations.components.connectors.weave.weave\_connector

<a id="haystack_integrations.components.connectors.weave.weave_connector.WeaveConnector"></a>

### WeaveConnector

Collects traces from your pipeline and sends them to Weights & Biases.

Add this component to your pipeline to integrate with the Weights & Biases Weave framework for tracing and
monitoring your pipeline components.

Note that you need to have the `WANDB_API_KEY` environment variable set to your Weights & Biases API key.

NOTE: If you don't have a Weights & Biases account it will interactively ask you to set one and your input
will then be stored in ~/.netrc

In addition, you need to set the `HAYSTACK_CONTENT_TRACING_ENABLED` environment variable to `true` in order to
enable Haystack tracing in your pipeline.

To use this connector simply add it to your pipeline without any connections, and it will automatically start
sending traces to Weights & Biases.

**Example**:

```python
import os

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.connectors import WeaveConnector

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

pipe = Pipeline()
pipe.add_component("prompt_builder", ChatPromptBuilder())
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-3.5-turbo"))
pipe.connect("prompt_builder.prompt", "llm.messages")

connector = WeaveConnector(pipeline_name="test_pipeline")
pipe.add_component("weave", connector)

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
```

  You should then head to `https://wandb.ai/<user_name>/projects` and see the complete trace for your pipeline under
  the pipeline name you specified, when creating the `WeaveConnector`

<a id="haystack_integrations.components.connectors.weave.weave_connector.WeaveConnector.__init__"></a>

#### WeaveConnector.\_\_init\_\_

```python
def __init__(pipeline_name: str,
             weave_init_kwargs: Optional[dict[str, Any]] = None) -> None
```

Initialize WeaveConnector.

**Arguments**:

- `pipeline_name`: The name of the pipeline you want to trace.
- `weave_init_kwargs`: Additional arguments to pass to the WeaveTracer client.

<a id="haystack_integrations.components.connectors.weave.weave_connector.WeaveConnector.warm_up"></a>

#### WeaveConnector.warm\_up

```python
def warm_up() -> None
```

Initialize the WeaveTracer.

<a id="haystack_integrations.components.connectors.weave.weave_connector.WeaveConnector.to_dict"></a>

#### WeaveConnector.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with all the necessary information to recreate this component.

<a id="haystack_integrations.components.connectors.weave.weave_connector.WeaveConnector.from_dict"></a>

#### WeaveConnector.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "WeaveConnector"
```

Deserializes the component from a dictionary.

**Arguments**:

- `data`: Dictionary to deserialize from.

**Returns**:

Deserialized component.

<a id="haystack_integrations.tracing.weave.tracer"></a>

## Module haystack\_integrations.tracing.weave.tracer

<a id="haystack_integrations.tracing.weave.tracer.WeaveSpan"></a>

### WeaveSpan

A bridge between Haystack's Span interface and Weave's Call object.

Stores metadata about a component execution and its inputs and outputs, and manages the attributes/tags
that describe the operation.

<a id="haystack_integrations.tracing.weave.tracer.WeaveSpan.set_tag"></a>

#### WeaveSpan.set\_tag

```python
def set_tag(key: str, value: Any) -> None
```

Set a tag by adding it to the call's inputs.

**Arguments**:

- `key`: The tag key.
- `value`: The tag value.

<a id="haystack_integrations.tracing.weave.tracer.WeaveSpan.raw_span"></a>

#### WeaveSpan.raw\_span

```python
def raw_span() -> Any
```

Access to the underlying Weave Call object.

<a id="haystack_integrations.tracing.weave.tracer.WeaveSpan.get_correlation_data_for_logs"></a>

#### WeaveSpan.get\_correlation\_data\_for\_logs

```python
def get_correlation_data_for_logs() -> dict[str, Any]
```

Correlation data for logging.

<a id="haystack_integrations.tracing.weave.tracer.WeaveSpan.set_content_tag"></a>

#### WeaveSpan.set\_content\_tag

```python
def set_content_tag(key: str, value: Any) -> None
```

Set a single tag containing content information.

Content is sensitive information such as
- the content of a query
- the content of a document
- the content of an answer

By default, this behavior is disabled. To enable it
- set the environment variable `HAYSTACK_CONTENT_TRACING_ENABLED` to `true` or
- override the `set_content_tag` method in a custom tracer implementation.

**Arguments**:

- `key`: the name of the tag.
- `value`: the value of the tag.

<a id="haystack_integrations.tracing.weave.tracer.WeaveTracer"></a>

### WeaveTracer

Implements a Haystack's Tracer to make an interface with Weights and Bias Weave.

It's responsible for creating and managing Weave calls, and for converting Haystack spans
to Weave spans. It creates spans for each Haystack component run.

<a id="haystack_integrations.tracing.weave.tracer.WeaveTracer.__init__"></a>

#### WeaveTracer.\_\_init\_\_

```python
def __init__(project_name: str, **weave_init_kwargs: Any) -> None
```

Initialize the WeaveTracer.

**Arguments**:

- `project_name`: The name of the project to trace, this is will be the name appearing in Weave project.
- `weave_init_kwargs`: Additional arguments to pass to the Weave client.

<a id="haystack_integrations.tracing.weave.tracer.WeaveTracer.current_span"></a>

#### WeaveTracer.current\_span

```python
def current_span() -> Optional[Span]
```

Get the current active span.

<a id="haystack_integrations.tracing.weave.tracer.WeaveTracer.trace"></a>

#### WeaveTracer.trace

```python
@contextlib.contextmanager
def trace(operation_name: str,
          tags: Optional[dict[str, Any]] = None,
          parent_span: Optional[WeaveSpan] = None) -> Iterator[WeaveSpan]
```

A context manager that creates and manages spans for tracking operations in Weights & Biases Weave.

It has two main workflows:

A) For regular operations (operation_name != "haystack.component.run"):
    Creates a Weave Call immediately
    Creates a WeaveSpan with this call
    Sets any provided tags
    Yields the span for use in the with block
    When the block ends, updates the call with pipeline output data

B) For component runs (operation_name == "haystack.component.run"):
    Creates a WeaveSpan WITHOUT a call initially (deferred creation)
    Sets any provided tags
    Yields the span for use in the with block
    Creates the actual Weave Call only at the end, when all component information is available
    Updates the call with component output data

This distinction is important because Weave's calls can't be updated once created, but the content
tags are only set on the Span at a later stage. To get the inputs on call creation, we need to create
the call after we yield the span.
