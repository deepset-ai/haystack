from haystack.tracing.tracer import (  # noqa: I001 (otherwise we end up with partial imports)
    Span,
    Tracer,
    auto_enable_tracing,
    disable_tracing,
    enable_tracing,
    is_tracing_enabled,
    tracer,
)
from haystack.tracing.opentelemetry import OpenTelemetryTracer
