import contextlib
from typing import Optional, Dict, Any, Iterator

from haystack.lazy_imports import LazyImport
from haystack.tracing import Tracer, Span
from haystack.tracing import utils as tracing_utils

with LazyImport("Run 'pip install ddtrace'") as ddtrace_import:
    import ddtrace


class DatadogSpan(Span):
    def __init__(self, span: "ddtrace.Span") -> None:
        self._span = span

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.set_tag(key, coerced_value)

    def raw_span(self) -> Any:
        return self._span

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        raw_span = self.raw_span()
        if not raw_span:
            return {}

        # https://docs.datadoghq.com/tracing/other_telemetry/connect_logs_and_traces/python/#no-standard-library-logging
        trace_id, span_id = (str((1 << 64) - 1 & raw_span.trace_id), raw_span.span_id)

        return {
            "dd.trace_id": trace_id,
            "dd.span_id": span_id,
            "dd.service": ddtrace.config.service or "",
            "dd.env": ddtrace.config.env or "",
            "dd.version": ddtrace.config.version or "",
        }


class DatadogTracer(Tracer):
    def __init__(self, tracer: "ddtrace.Tracer") -> None:
        ddtrace_import.check()
        self._tracer = tracer

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        with self._tracer.trace(operation_name) as span:
            span = DatadogSpan(span)
            if tags:
                span.set_tags(tags)

            yield span

    def current_span(self) -> Optional[Span]:
        current_span = self._tracer.current_span()
        if current_span is None:
            return None

        return DatadogSpan(current_span)
