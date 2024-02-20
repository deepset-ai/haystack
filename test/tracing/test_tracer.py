from unittest.mock import Mock

from haystack.tracing.tracer import (
    get_tracer,
    NullTracer,
    NullSpan,
    enable_tracing,
    Tracer,
    disable_tracing,
    is_tracing_enabled,
)


class TestNullTracer:
    def test_tracing(self) -> None:
        assert isinstance(get_tracer(), NullTracer)

        # None of this raises
        with get_tracer().trace("operation", {"key": "value"}) as span:
            span.set_tag("key", "value")
            span.set_tags({"key": "value"})

        assert isinstance(get_tracer().current_span(), NullSpan)
        assert isinstance(get_tracer().current_raw_span(), NullSpan)


class TestConfigureTracer:
    def test_enable_tracer(self) -> None:
        my_tracer = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test

        enable_tracing(my_tracer)

        assert get_tracer() is my_tracer
        assert is_tracing_enabled()

    def test_disable_tracing(self) -> None:
        my_tracker = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test

        enable_tracing(my_tracker)
        assert get_tracer() is my_tracker

        disable_tracing()
        assert isinstance(get_tracer(), NullTracer)
        assert is_tracing_enabled() is False
