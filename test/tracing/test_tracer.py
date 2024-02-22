from unittest.mock import Mock

from haystack.tracing.tracer import (
    NullTracer,
    NullSpan,
    enable_tracing,
    Tracer,
    disable_tracing,
    is_tracing_enabled,
    ProxyTracer,
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
