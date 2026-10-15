---
title: "Rhesis"
id: integrations-rhesis
description: "Rhesis integration for Haystack"
slug: "/integrations-rhesis"
---


## haystack_integrations.components.connectors.rhesis.rhesis_connector

### RhesisConnector

Connects Haystack to [Rhesis](https://rhesis.ai) for OpenTelemetry-based tracing of pipelines.

Add this component to a pipeline without connecting it to other components. It enables tracing
for all pipeline operations when Haystack tracing is active.

**Environment Configuration:**

- `RHESIS_API_KEY`: Required API key for trace ingestion.
- `RHESIS_BASE_URL`: Backend URL (default `http://localhost:8080` for local development).
- `RHESIS_PROJECT_ID`: Optional project identifier (resolved from the API key when omitted).
- `RHESIS_ENVIRONMENT`: Deployment environment label (default `development`).
- `RHESIS_FRONTEND_URL`: Optional frontend URL used to build `trace_url` deep links.
- `HAYSTACK_CONTENT_TRACING_ENABLED`: Must be `"true"` **before importing Haystack** to
  capture input/output on spans.
- `HAYSTACK_RHESIS_ENFORCE_FLUSH`: When `"true"` (default), exports once per pipeline run,
  as the root span closes. Set to `"false"` to leave exporting to the batch processor and
  flush on shutdown instead.

Example shutdown flush for FastAPI:

```python
from haystack.tracing import tracer

@app.on_event("shutdown")
async def shutdown_event():
    tracer.actual_tracer.flush()
```

#### __init__

```python
__init__(
    name: str,
    api_key: Secret | None = Secret.from_env_var("RHESIS_API_KEY"),
    base_url: str | None = None,
    project_id: str | None = None,
    environment: str | None = None,
    frontend_url: str | None = None,
    span_handler: SpanHandler | None = None,
) -> None
```

Initialize the RhesisConnector component.

**Parameters:**

- **name** (<code>str</code>) – Trace name shown in the Rhesis UI.
- **api_key** (<code>Secret | None</code>) – Rhesis API key. Defaults to `RHESIS_API_KEY`.
- **base_url** (<code>str | None</code>) – Rhesis backend base URL. Defaults to `RHESIS_BASE_URL` or
  `http://localhost:8080`.
- **project_id** (<code>str | None</code>) – Rhesis project ID. Defaults to `RHESIS_PROJECT_ID`.
- **environment** (<code>str | None</code>) – Environment label. Defaults to `RHESIS_ENVIRONMENT` or
  `development`.
- **frontend_url** (<code>str | None</code>) – Frontend base URL for `trace_url`. Defaults to `RHESIS_FRONTEND_URL`.
- **span_handler** (<code>SpanHandler | None</code>) – Optional custom span handler. Uses :class:`DefaultSpanHandler` when omitted.

**Raises:**

- <code>ValueError</code> – If no API key resolves. A component the user explicitly added to a
  pipeline should say so rather than silently trace nothing — but it does mean
  `Pipeline.from_dict` on a YAML containing this component needs credentials present.
  :class:`~haystack_integrations.tracing.rhesis.RhesisTracing` deliberately does the
  opposite and degrades to a no-op, because there the caller did not put tracing in the
  data path.

#### run

```python
run(invocation_context: dict[str, Any] | None = None) -> dict[str, str]
```

Run the connector and return trace metadata.

The context applies to the pipeline run that invoked this component and no other. The
ContextVar it is written to is set by `RhesisTracer.trace` when the run's root span opens
and restored when that span closes, so this write lands inside that scope and cannot outlive
the run — which is why the context is only honoured when a root span is open. Outside one
there is nothing to scope it to and nothing to stamp it on, so it is ignored rather than left
behind for the next caller to inherit. To attach metadata to work that is not a pipeline run
— a standalone `Agent`, say — wrap the call in
:func:`~haystack_integrations.tracing.rhesis.rhesis_invocation_context` instead, which scopes
the value to its own block.

**Parameters:**

- **invocation_context** (<code>dict\[str, Any\] | None</code>) – Optional key-value metadata attached to the root trace
  (session, test run identifiers, tags, etc.).

**Returns:**

- <code>dict\[str, str\]</code> – Dictionary with `name`, `trace_url`, and `trace_id`.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

Records the arguments as they were passed, not as they were resolved: anything left to the
environment stays `None` so that deserializing on another machine resolves it there. This
mirrors how `Secret.from_env_var` serializes a reference rather than the secret's value.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> RhesisConnector
```

Deserialize this component from a dictionary.

## haystack_integrations.tracing.rhesis.conversation

Conversation-aware tracing for applications that drive Haystack from their own loop.

### ConversationTurn

A single conversation turn, yielded by :meth:`RhesisTracing.turn`.

Assign :attr:`output` with the reply the user actually sees. Only the application knows
what that is — it may be a tool result or a value held in agent state rather than the last
assistant message — so it cannot be inferred from the span tree.

#### span

```python
span: Span | None
```

The underlying OTel span, or `None` when tracing is disabled.

#### output

```python
output: str
```

The reply recorded for this turn.

### RhesisTracing

Enable Rhesis tracing for an application that runs Haystack from its own loop.

:class:`RhesisConnector` covers the common case: add it to a pipeline and every run is
traced. An application that owns its loop — a chat REPL, a batch script, a server handling
one turn per request — needs two things a component inside the pipeline cannot provide:
tracing switched on without a pipeline to attach to, and a span wrapping a whole pipeline
run so a conversation turn has a root of its own.

Without that root span, the Haystack pipeline span claims the turn and reports the
serialized pipeline input and output as the conversation text.

`HAYSTACK_CONTENT_TRACING_ENABLED` must still be set to `"true"` before Haystack is
imported, exactly as when using the connector.

### Usage example

```python
import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack_integrations.tracing.rhesis import RhesisTracing

tracing = RhesisTracing("My Assistant")
tracing.start_conversation("conversation-1")

for message in ["Hello", "Tell me more"]:
    with tracing.turn(message) as turn:
        result = pipeline.run(...)
        turn.output = result["llm"]["replies"][0].text

tracing.flush()
```

#### __init__

```python
__init__(
    name: str,
    *,
    enabled: bool = True,
    turn_span_name: str = DEFAULT_TURN_SPAN_NAME,
    **connector_kwargs: Any
) -> None
```

Enable tracing, or fall back to a no-op when Rhesis is not configured.

Construction never raises on a missing or rejected configuration: an application should
run untraced rather than fail to start. Check :attr:`enabled` to report it.

This is the opposite of :class:`~haystack_integrations.components.connectors.rhesis.RhesisConnector`,
which raises when no API key resolves, and deliberately so. The connector is a component the
user put in a pipeline; failing loudly there is the honest signal that the thing they wired
up will not do its job. Here tracing wraps an application's own loop and is not in its data
path, so the same failure should cost the application nothing.

**Parameters:**

- **name** (<code>str</code>) – Trace name shown in the Rhesis UI.
- **enabled** (<code>bool</code>) – Set to `False` to build a no-op instance, so an application can gate
  tracing on its own policy without branching around every call.
- **turn_span_name** (<code>str</code>) – Span name for each conversation turn root.
- **connector_kwargs** (<code>Any</code>) – Forwarded to :class:`RhesisConnector` (`api_key`,
  `base_url`, `project_id`, `environment`, `frontend_url`, `span_handler`).

#### enabled

```python
enabled: bool
```

Whether tracing was successfully enabled.

#### start_conversation

```python
start_conversation(conversation_id: str, **invocation_context: Any) -> None
```

Group the turns that follow into one conversation, sharing one trace.

Calling this again starts a new conversation: the next turn opens a new trace and later
turns join it.

**Parameters:**

- **conversation_id** (<code>str</code>) – Identifier grouping the turns, shown as the conversation in Rhesis.
- **invocation_context** (<code>Any</code>) – Extra metadata for the root span, as
  :meth:`RhesisConnector.run` accepts (test run identifiers, tags, …).

#### turn

```python
turn(user_input: str) -> Iterator[ConversationTurn]
```

Open the root span for one conversation turn.

Run the turn's work inside the block and assign the reply to
:attr:`ConversationTurn.output`. Every turn after the first joins the first one's trace,
so a conversation reads as one trace rather than one per exchange.

Yields an inert turn when tracing is disabled, so callers need no branching.

**Parameters:**

- **user_input** (<code>str</code>) – The user's message, recorded as the turn's conversation input.

#### flush

```python
flush() -> None
```

Flush pending spans. Call before exit; batched spans are otherwise lost.

## haystack_integrations.tracing.rhesis.tracer

Rhesis tracing bridge for Haystack.

Set `HAYSTACK_CONTENT_TRACING_ENABLED=true` before importing Haystack to capture
input/output content on spans.

### rhesis_invocation_context

```python
rhesis_invocation_context(
    invocation_context: dict[str, Any] | None = None,
) -> Iterator[None]
```

Attach Rhesis session/test metadata for the current async task or thread.

### RhesisTelemetry

Thin wrapper around the OTel provider used by the Haystack integration.

#### flush

```python
flush() -> None
```

Flush pending spans to the Rhesis backend.

### resolve_frontend_url

```python
resolve_frontend_url(base_url: str, frontend_url: str | None) -> str
```

Resolve the Rhesis frontend base URL for trace deep links.

Only the two well-known deployments are derived from `base_url`. Any other backend returns an
empty string — and therefore an empty `trace_url` — unless `RHESIS_FRONTEND_URL` is set.

**Parameters:**

- **base_url** (<code>str</code>) – The Rhesis backend base URL.
- **frontend_url** (<code>str | None</code>) – An explicit frontend origin, which always wins when given.

**Returns:**

- <code>str</code> – The frontend origin without a trailing slash, or `""` when it cannot be derived.

### build_trace_url

```python
build_trace_url(
    frontend_url: str, trace_id: str, project_id: str | None
) -> str
```

Build a frontend deep link for the given trace.

### RhesisSpan

Bases: <code>Span</code>

Bridge between Haystack's span API and OpenTelemetry spans for Rhesis.

#### set_tag

```python
set_tag(key: str, value: Any) -> None
```

Set a generic tag for this span.

#### set_content_tag

```python
set_content_tag(key: str, value: Any) -> None
```

Set a content-specific tag for this span when content tracing is enabled.

#### raw_span

```python
raw_span() -> trace.Span
```

Return the underlying OpenTelemetry span instance.

#### close

```python
close(exc_info: tuple[Any, Any, Any] | None = None) -> None
```

End the underlying OpenTelemetry span.

**Parameters:**

- **exc_info** (<code>tuple\[Any, Any, Any\] | None</code>) – The `sys.exc_info()` triple when the span is closing because of an
  exception, so the context manager sees it; `None` on the success path.

#### get_data

```python
get_data() -> dict[str, Any]
```

Return the raw Haystack tag data collected for this span.

#### get_correlation_data_for_logs

```python
get_correlation_data_for_logs() -> dict[str, Any]
```

Return trace and span identifiers for log correlation.

#### set_tags

```python
set_tags(tags: dict[str, Any]) -> None
```

Set multiple tags on this span.

### SpanContext

Context for creating spans in Rhesis.

### SpanHandler

Bases: <code>ABC</code>

Extension point for customizing Rhesis span creation and enrichment.

#### init_tracer

```python
init_tracer(tracer: RhesisTelemetry) -> None
```

Initialize with the Rhesis telemetry wrapper.

#### create_span

```python
create_span(context: SpanContext) -> RhesisSpan
```

Create a span of appropriate type based on the context.

#### handle

```python
handle(span: RhesisSpan, component_type: str | None) -> None
```

Process a span after component execution.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> SpanHandler
```

Deserialize a SpanHandler from a dictionary.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serialize this SpanHandler to a dictionary.

### DefaultSpanHandler

Bases: <code>SpanHandler</code>

Default Rhesis tracing behavior for Haystack pipelines.

#### create_span

```python
create_span(context: SpanContext) -> RhesisSpan
```

Create a Rhesis span based on the given Haystack context.

#### handle

```python
handle(span: RhesisSpan, component_type: str | None) -> None
```

Process and enrich a span after component execution.

### RhesisTracer

Bases: <code>Tracer</code>

Haystack tracer implementation that exports spans to Rhesis via OpenTelemetry.

#### __init__

```python
__init__(
    telemetry: RhesisTelemetry,
    name: str = "Haystack",
    span_handler: SpanHandler | None = None,
) -> None
```

Initialize a RhesisTracer instance.

**Parameters:**

- **telemetry** (<code>RhesisTelemetry</code>) – Configured Rhesis OpenTelemetry telemetry wrapper.
- **name** (<code>str</code>) – Trace name shown in the Rhesis UI.
- **span_handler** (<code>SpanHandler | None</code>) – Custom handler for span creation and enrichment.

#### telemetry

```python
telemetry: RhesisTelemetry
```

The Rhesis OTel provider and tracer backing this tracer.

Public because the provider is private to this tracer: it is not installed as the
OpenTelemetry global, so anything that needs to open a span destined for Rhesis — the
conversation turn spans in :class:`~haystack_integrations.tracing.rhesis.RhesisTracing`, or a
custom :class:`SpanHandler` — has to reach it through here rather than through
`trace.get_tracer()`.

#### trace

```python
trace(
    operation_name: str,
    tags: dict[str, Any] | None = None,
    parent_span: Span | None = None,
) -> Iterator[Span]
```

Create and manage a tracing span as a context manager.

#### flush

```python
flush() -> None
```

Flush all pending spans to Rhesis.

#### current_span

```python
current_span() -> Span | None
```

Return the current active span.

#### get_trace_url

```python
get_trace_url() -> str
```

Return the frontend URL for the current trace, when available.

#### get_trace_id

```python
get_trace_id() -> str
```

Return the trace ID of the root span currently open in this context.
