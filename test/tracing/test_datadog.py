# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import functools
import json
from typing import List, Dict

import ddtrace
import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from haystack.tracing.datadog import DatadogTracer


@pytest.fixture()
def datadog_tracer(monkeypatch: MonkeyPatch) -> ddtrace.Tracer:
    # For the purpose of the tests we want to use the log writer
    monkeypatch.setattr(ddtrace.Tracer, ddtrace.Tracer._use_log_writer.__name__, lambda *_: True)

    tracer = ddtrace.Tracer()

    return tracer


def get_traces_from_console(capfd: CaptureFixture) -> List[Dict]:
    output = capfd.readouterr().out
    parsed = json.loads(output)
    nested_traces = parsed["traces"]
    flattened = list(functools.reduce(lambda x, y: x + y, nested_traces, []))

    return flattened


class TestDatadogTracer:
    def test_opentelemetry_tracer(self, datadog_tracer: ddtrace.Tracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", "value")

        traces = get_traces_from_console(capfd)
        assert len(traces) == 1

        trace = traces[0]

        assert trace["name"] == "test"

    def test_tagging(self, datadog_tracer: ddtrace.Tracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test", tags={"key1": "value1"}) as span:
            span.set_tag("key2", "value2")

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key1"] == "value1"
        assert spans[0]["meta"]["key2"] == "value2"

    def test_current_span(self, datadog_tracer: ddtrace.Tracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test"):
            current_span = tracer.current_span()
            assert tracer.current_span() is not None

            current_span.set_tag("key1", "value1")

            raw_span = current_span.raw_span()
            assert raw_span is not None
            assert isinstance(raw_span, ddtrace.Span)

            raw_span.set_tag("key2", "value2")

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key1"] == "value1"
        assert spans[0]["meta"]["key2"] == "value2"

    def test_tracing_complex_values(self, datadog_tracer: ddtrace.Tracer, capfd: CaptureFixture) -> None:
        tracer = DatadogTracer(datadog_tracer)

        with tracer.trace("test") as span:
            span.set_tag("key", {"a": 1, "b": [2, 3, 4]})

        spans = get_traces_from_console(capfd)
        assert len(spans) == 1
        assert spans[0]["meta"]["key"] == '{"a": 1, "b": [2, 3, 4]}'

    def test_get_log_correlation_info(self, datadog_tracer: ddtrace.Tracer) -> None:
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
