import contextlib
from typing import Any, Dict, Iterator, Optional

from haystack.lazy_imports import LazyImport
from haystack.tracing import Span, Tracer
from haystack.tracing import utils as tracing_utils

with LazyImport("Run 'pip install opentelemetry-sdk'") as opentelemetry_import:
    import opentelemetry
    import opentelemetry.trace


class OpenTelemetrySpan(Span):
    def __init__(self, span: "opentelemetry.trace.Span") -> None:
        self._span = span

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.set_attribute(key, coerced_value)

    def raw_span(self) -> Any:
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        span_context = self._span.get_span_context()
        return {"trace_id": span_context.trace_id, "span_id": span_context.span_id}


class OpenTelemetryTracer(Tracer):
    def __init__(self, tracer: "opentelemetry.trace.Tracer") -> None:
        opentelemetry_import.check()
        self._tracer = tracer

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        with self._tracer.start_as_current_span(operation_name) as raw_span:
            span = OpenTelemetrySpan(raw_span)
            if tags:
                span.set_tags(tags)

            yield span

    def current_span(self) -> Optional[Span]:
        current_span = opentelemetry.trace.get_current_span()
        if isinstance(current_span, opentelemetry.trace.NonRecordingSpan):
            return None

        return OpenTelemetrySpan(current_span)
