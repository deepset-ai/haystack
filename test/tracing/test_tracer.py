import builtins
import sys
from unittest.mock import Mock

import ddtrace
import opentelemetry.trace
import pytest
from _pytest.monkeypatch import MonkeyPatch
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from haystack.tracing.datadog import DatadogTracer
from haystack.tracing.opentelemetry import OpenTelemetryTracer
from haystack.tracing.tracer import (
    NullTracer,
    NullSpan,
    enable_tracing,
    Tracer,
    disable_tracing,
    is_tracing_enabled,
    ProxyTracer,
    auto_enable_tracing,
    tracer,
)
from test.tracing.utils import SpyingTracer


class TestNullTracer:
    def test_tracing(self) -> None:
        assert isinstance(tracer.actual_tracer, NullTracer)

        # None of this raises
        with tracer.trace("operation", {"key": "value"}) as span:
            span.set_tag("key", "value")
            span.set_tags({"key": "value"})

        assert isinstance(tracer.current_span(), NullSpan)
        assert isinstance(tracer.current_span().raw_span(), NullSpan)


class TestProxyTracer:
    def test_tracing(self) -> None:
        spying_tracer = SpyingTracer()
        my_tracer = ProxyTracer(provided_tracer=spying_tracer)

        enable_tracing(spying_tracer)

        with my_tracer.trace("operation", {"key": "value"}) as span:
            span.set_tag("key", "value")
            span.set_tags({"key2": "value2"})

        assert len(spying_tracer.spans) == 1
        assert spying_tracer.spans[0].operation_name == "operation"
        assert spying_tracer.spans[0].tags == {"key": "value", "key2": "value2"}


class TestConfigureTracer:
    def test_enable_tracer(self) -> None:
        my_tracer = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test

        enable_tracing(my_tracer)

        assert isinstance(tracer, ProxyTracer)
        assert tracer.actual_tracer is my_tracer
        assert is_tracing_enabled()

    def test_disable_tracing(self) -> None:
        my_tracker = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test

        enable_tracing(my_tracker)
        assert tracer.actual_tracer is my_tracker

        disable_tracing()
        assert isinstance(tracer.actual_tracer, NullTracer)
        assert is_tracing_enabled() is False


class TestAutoEnableTracer:
    @pytest.fixture()
    def configured_opentelemetry_tracing(self) -> None:
        resource = Resource(attributes={SERVICE_NAME: "haystack-testing"})

        traceProvider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(InMemorySpanExporter())
        traceProvider.add_span_processor(processor)

        # We can't uset `set_tracer_provider` here, because opentelemetry has a lock to only set it once
        opentelemetry.trace._TRACER_PROVIDER = traceProvider

        yield

        # unfortunately, there's no cleaner way to reset the global tracer provider
        opentelemetry.trace._TRACER_PROVIDER = None
        disable_tracing()

    @pytest.fixture()
    def uninstalled_ddtrace_package(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(ddtrace.tracer, "enabled", False)

    def test_skip_auto_enable_tracer_if_already_configured(self) -> None:
        my_tracker = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test
        enable_tracing(my_tracker)

        auto_enable_tracing()

        assert tracer.actual_tracer is my_tracker

    def test_skip_auto_enable_if_tracing_disabled_via_env(
        self, monkeypatch: MonkeyPatch, configured_opentelemetry_tracing: None
    ) -> None:
        monkeypatch.setenv("HAYSTACK_AUTO_TRACE_ENABLED", "false")

        old_tracer = tracer.actual_tracer

        auto_enable_tracing()

        assert tracer.actual_tracer is old_tracer

    def test_enable_opentelemetry_tracer(self, configured_opentelemetry_tracing: None) -> None:
        auto_enable_tracing()

        activated_tracer = tracer.actual_tracer
        assert isinstance(activated_tracer, OpenTelemetryTracer)
        assert is_tracing_enabled()

    def test_skip_enable_opentelemetry_tracer_if_import_error(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "opentelemetry", raising=False)
        monkeypatch.setattr(builtins, "__import__", Mock(side_effect=ImportError))
        auto_enable_tracing()

        activated_tracer = tracer.actual_tracer
        assert isinstance(activated_tracer, NullTracer)
        assert not is_tracing_enabled()

    def test_skip_add_datadog_tracer_if_import_error(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "ddtrace", raising=False)
        monkeypatch.setattr(builtins, "__import__", Mock(side_effect=ImportError))
        auto_enable_tracing()

        activated_tracer = tracer.actual_tracer
        assert isinstance(activated_tracer, NullTracer)
        assert not is_tracing_enabled()

    def test_add_datadog_tracer(self) -> None:
        auto_enable_tracing()

        activated_tracer = tracer.actual_tracer
        assert isinstance(activated_tracer, DatadogTracer)
        assert is_tracing_enabled()
