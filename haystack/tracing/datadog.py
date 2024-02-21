import contextlib
from typing import Optional, Dict, Any, Iterator

from haystack.lazy_imports import LazyImport
from haystack.tracing import Tracer, Span
from haystack.tracing import utils as tracing_utils

with LazyImport("Run 'pip install ddtrace'") as ddtrace_import:
    import ddtrace


class DatadogSpan(Span):
    def __init__(self, span: ddtrace.Span) -> None:
        self._span = span

    def set_tag(self, key: str, value: Any) -> None:
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.set_tag(key, coerced_value)

    def raw_span(self) -> Any:
        return self._span


class DatadogTracer(Tracer):
    def __init__(self, tracer: ddtrace.Tracer) -> None:
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
