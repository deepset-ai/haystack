# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import opentelemetry.trace
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from opentelemetry.sdk.trace import TracerProvider

from haystack.tracing.opentelemetry import OpenTelemetryTracer


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def opentelemetry_tracer(span_exporter: InMemorySpanExporter) -> opentelemetry.trace.Tracer:
    # Service name is required for most backends
    resource = Resource(attributes={SERVICE_NAME: "haystack-testing"})

    traceProvider = TracerProvider(resource=resource)
    processor = SimpleSpanProcessor(span_exporter)
    traceProvider.add_span_processor(processor)

    return traceProvider.get_tracer("my_test")


class TestOpenTelemetryTracer:
    def test_opentelemetry_tracer(
        self, opentelemetry_tracer: opentelemetry.trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        tracer = OpenTelemetryTracer(opentelemetry_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", "value")

    def test_tagging(
        self, opentelemetry_tracer: opentelemetry.trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        tracer = OpenTelemetryTracer(opentelemetry_tracer)

        with tracer.trace("test", tags={"key1": "value1"}) as span:
            span.set_tag("key2", "value2")

        spans = list(span_exporter.get_finished_spans())
        assert len(spans) == 1
        assert spans[0].attributes == {"key1": "value1", "key2": "value2"}

    def test_current_span(
        self, opentelemetry_tracer: opentelemetry.trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        tracer = OpenTelemetryTracer(opentelemetry_tracer)
        with tracer.trace("test"):
            current_span = tracer.current_span()
            assert tracer.current_span() is not None

            current_span.set_tag("key1", "value1")

            raw_span = current_span.raw_span()
            assert raw_span is not None
            assert isinstance(raw_span, opentelemetry.trace.Span)

            raw_span.set_attribute("key2", "value2")

        spans = list(span_exporter.get_finished_spans())
        assert len(spans) == 1
        assert spans[0].attributes == {"key1": "value1", "key2": "value2"}

    def test_tracing_complex_values(
        self, opentelemetry_tracer: opentelemetry.trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        tracer = OpenTelemetryTracer(opentelemetry_tracer)
        with tracer.trace("test") as span:
            span.set_tag("key", {"a": 1, "b": [2, 3, 4]})

        spans = list(span_exporter.get_finished_spans())
        assert len(spans) == 1
        assert spans[0].attributes == {"key": '{"a": 1, "b": [2, 3, 4]}'}

    def test_log_correlation_info(self, opentelemetry_tracer: opentelemetry.trace.Tracer) -> None:
        tracer = OpenTelemetryTracer(opentelemetry_tracer)
        with tracer.trace("test") as span:
            span.set_tag("key", "value")

        correlation_data = span.get_correlation_data_for_logs()
        assert correlation_data == {
            "trace_id": span.raw_span().get_span_context().trace_id,
            "span_id": span.raw_span().get_span_context().span_id,
        }
