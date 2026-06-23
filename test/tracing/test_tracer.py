# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from _pytest.monkeypatch import MonkeyPatch

from haystack.tracing.tracer import (
    HAYSTACK_CONTENT_TRACING_ENABLED_ENV_VAR,
    NullSpan,
    NullTracer,
    ProxyTracer,
    Tracer,
    auto_enable_tracing,
    disable_tracing,
    enable_tracing,
    is_tracing_enabled,
    tracer,
)
from test.tracing.utils import SpyingSpan, SpyingTracer


class TestNullTracer:
    def test_tracing(self) -> None:
        assert isinstance(tracer.actual_tracer, NullTracer)

        # None of this raises
        with tracer.trace("operation", {"key": "value"}, parent_span=None) as span:
            span.set_tag("key", "value")
            span.set_tags({"key": "value"})

        assert isinstance(tracer.current_span(), NullSpan)
        current_span = tracer.current_span()
        assert current_span is not None
        assert isinstance(current_span.raw_span(), NullSpan)


class TestProxyTracer:
    def test_tracing(self) -> None:
        spying_tracer = SpyingTracer()
        my_tracer = ProxyTracer(provided_tracer=spying_tracer)

        parent_span = Mock(spec=SpyingSpan)
        with my_tracer.trace("operation", {"key": "value"}, parent_span=parent_span) as span:
            span.set_tag("key", "value")
            span.set_tags({"key2": "value2"})

        assert len(spying_tracer.spans) == 1
        assert spying_tracer.spans[0].operation_name == "operation"
        assert spying_tracer.spans[0].parent_span == parent_span
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
    def test_skip_auto_enable_tracer_if_already_configured(self) -> None:
        my_tracker = Mock(spec=Tracer)  # anything else than `NullTracer` works for this test
        enable_tracing(my_tracker)

        auto_enable_tracing()

        assert tracer.actual_tracer is my_tracker

        disable_tracing()

    def test_skip_auto_enable_if_tracing_disabled_via_env(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("HAYSTACK_AUTO_TRACE_ENABLED", "false")

        old_tracer = tracer.actual_tracer

        auto_enable_tracing()

        assert tracer.actual_tracer is old_tracer

    def test_auto_enable_tracing_does_not_configure_a_backend(self) -> None:
        # Haystack no longer ships a built-in tracing backend, so `auto_enable_tracing` must not enable one.
        disable_tracing()

        auto_enable_tracing()

        assert is_tracing_enabled() is False


class TestTracingContent:
    def test_set_content_tag_with_enabled_content_tracing(self, spying_tracer: SpyingTracer) -> None:
        # SpyingTracer supports content tracing by default

        enable_tracing(spying_tracer)
        with tracer.trace("test") as span:
            span.set_content_tag("my_content", "my_content")

        assert len(spying_tracer.spans) == 1
        span = spying_tracer.spans[0]
        assert span.tags == {"my_content": "my_content"}

    def test_set_content_tag_when_disabled_via_env_variable(self, monkeypatch: MonkeyPatch) -> None:
        # we test if content tracing is disabled when the env variable is set to false
        monkeypatch.setenv(HAYSTACK_CONTENT_TRACING_ENABLED_ENV_VAR, "false")

        proxy_tracer = ProxyTracer(provided_tracer=SpyingTracer())

        assert proxy_tracer.is_content_tracing_enabled is False
