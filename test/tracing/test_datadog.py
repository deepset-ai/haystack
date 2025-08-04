# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import functools
import json

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from ddtrace.trace import Span as ddSpan
from ddtrace.trace import Tracer as ddTracer

from haystack.tracing.datadog import DatadogTracer


@pytest.fixture()
def datadog_tracer(monkeypatch: MonkeyPatch) -> ddTracer:
    # For the purpose of the tests we want to use the log writer.
    # We simulate being in AWS Lambda, where the log writer is active.
    # See https://github.com/DataDog/dd-trace-py/blob/ae4c189ebf8e539f39905f21c7918cc19de69d13/ddtrace/internal/writer/writer.py#L680
    # for more details.
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "test-function")

    tracer = ddTracer()

    return tracer


def get_traces_from_console(capfd: CaptureFixture) -> list[dict]:
    output = capfd.readouterr().out
    parsed = json.loads(output)
    nested_traces = parsed["traces"]
    flattened = list(functools.reduce(lambda x, y: x + y, nested_traces, []))

    return flattened


class TestDatadogTracer:
    def test_opentelemetry_tracer(self, datadog_tracer: ddTracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        component_tags = {
            "haystack.component.name": "test_component",
            "haystack.component.type": "TestType",
            "haystack.component.input": {"input_key": "input_value"},
            "haystack.component.output": {"output_key": "output_value"},
        }

        with tracer.trace("haystack.component.run", tags=component_tags) as span:
            span.set_tag("key", "value")

        traces = get_traces_from_console(capfd)
        assert len(traces) == 1

        trace = traces[0]

        assert trace["name"] == "haystack.component.run"
        assert "test_component" in trace["resource"]
        assert "TestType" in trace["resource"]

    def test_tagging(self, datadog_tracer: ddTracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test", tags={"key1": "value1"}) as span:
            span.set_tag("key2", "value2")

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key1"] == "value1"
        assert spans[0]["meta"]["key2"] == "value2"

    def test_current_span(self, datadog_tracer: ddTracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test"):
            current_span = tracer.current_span()
            assert tracer.current_span() is not None

            current_span.set_tag("key1", "value1")

            raw_span = current_span.raw_span()
            assert raw_span is not None
            assert isinstance(raw_span, ddSpan)

            raw_span.set_tag("key2", "value2")

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key1"] == "value1"
        assert spans[0]["meta"]["key2"] == "value2"

    def test_tracing_complex_values(self, datadog_tracer: ddTracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", {"a": 1, "b": [2, 3, 4]})

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key"] == '{"a": 1, "b": [2, 3, 4]}'

    def test_get_log_correlation_info(self, datadog_tracer: ddTracer) -> None:
        tracer = DatadogTracer(datadog_tracer)
        with tracer.trace("test") as span:
            span.set_tag("key", "value")
            assert span.get_correlation_data_for_logs() == {
                "dd.trace_id": str((1 << 64) - 1 & span.raw_span().trace_id),
                "dd.span_id": span.raw_span().span_id,
                "dd.service": "",
                "dd.env": "",
                "dd.version": "",
            }
