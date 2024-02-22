import abc
import contextlib
import logging
import os
from typing import Dict, Any, Optional, Iterator

HAYSTACK_AUTO_TRACE_ENABLED_ENV_VAR = "HAYSTACK_AUTO_TRACE_ENABLED"

logger = logging.getLogger(__name__)


class Span(abc.ABC):
    """Interface for an instrumented operation."""

    @abc.abstractmethod
    def set_tag(self, key: str, value: Any) -> None:
        """Set a single tag on the span.

        Note that the value will be serialized to a string, so it's best to use simple types like strings, numbers, or
        booleans.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        pass

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set multiple tags on the span.

        :param tags: a mapping of tag names to tag values.
        """
        for key, value in tags.items():
            self.set_tag(key, value)

    def raw_span(self) -> Any:
        """Provides access to the underlying span object of the tracer.

        Use this if you need full access to the underlying span object.

        :return: The underlying span object.
        """
        return self


class Tracer(abc.ABC):
    """Interface for instrumenting code by creating and submitting spans."""

    @abc.abstractmethod
    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        """Trace the execution of a block of code.

        :param operation_name: the name of the operation being traced.
        :param tags: tags to apply to the newly created span.
        :return: the newly created span.
        """
        pass

    @abc.abstractmethod
    def current_span(self) -> Optional[Span]:
        """Returns the currently active span. If no span is active, returns `None`.

        :return: Currently active span or `None` if no span is active.
        """
        pass


class ProxyTracer(Tracer):
    """Container for the actual tracer instance.

    This eases
    - replacing the actual tracer instance without having to change the global tracer instance
    - implementing default behavior for the tracer
    """

    def __init__(self, provided_tracer: Tracer) -> None:
        self.actual_tracer: Tracer = provided_tracer

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        with self.actual_tracer.trace(operation_name, tags=tags) as span:
            yield span

    def current_span(self) -> Optional[Span]:
        return self.actual_tracer.current_span()


class NullSpan(Span):
    """A no-op implementation of the `Span` interface. This is used when tracing is disabled."""

    def set_tag(self, key: str, value: Any) -> None:
        pass


class NullTracer(Tracer):
    """A no-op implementation of the `Tracer` interface. This is used when tracing is disabled."""

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        yield NullSpan()

    def current_span(self) -> Optional[Span]:
        return NullSpan()


# We use the proxy pattern to allow for easy enabling and disabling of tracing without having to change the global
# tracer instance. That's especially convenient if users import the object directly
# (in that case we'd have to monkey-patch it in all of these modules).
tracer: ProxyTracer = ProxyTracer(provided_tracer=NullTracer())


def enable_tracing(provided_tracer: Tracer) -> None:
    """Enable tracing by setting the global tracer instance."""
    tracer.actual_tracer = provided_tracer


def disable_tracing() -> None:
    """Disable tracing by setting the global tracer instance to a no-op tracer."""
    tracer.actual_tracer = NullTracer()


def is_tracing_enabled() -> bool:
    """Return whether tracing is enabled."""
    return not isinstance(tracer.actual_tracer, NullTracer)


def auto_enable_tracing() -> None:
    """Auto-enable the right tracing backend.

    This behavior can be disabled by setting the environment variable `HAYSTACK_AUTO_TRACE_ENABLED` to `false`.
    Note that it will only work correctly if tracing was configured _before_ Haystack is imported.
    """
    if os.getenv(HAYSTACK_AUTO_TRACE_ENABLED_ENV_VAR, "true").lower() == "false":
        logger.info("Tracing disabled via '%s'", HAYSTACK_AUTO_TRACE_ENABLED_ENV_VAR)
        return

    if is_tracing_enabled():
        return  # tracing already enabled

    tracer = _auto_configured_opentelemetry_tracer()
    if tracer:
        enable_tracing(tracer)
        logger.info("Tracing enabled via '%s'", tracer.__class__.__name__)


def _auto_configured_opentelemetry_tracer() -> Optional[Tracer]:
    # we implement this here and not in the `opentelemetry` module to avoid import warnings when OpenTelemetry is not
    # installed
    try:
        import opentelemetry.trace

        # the safest way to check if tracing is enabled is to try to start a span and see if it's a no-op span
        # alternatively we could of course check `opentelemetry.trace._TRACER_PROVIDER`
        # but that's not part of the public API and could change in the future
        with opentelemetry.trace.get_tracer("haystack").start_as_current_span("haystack.tracing.auto_enable") as span:
            if isinstance(span, opentelemetry.trace.NonRecordingSpan):
                return None

            from haystack.tracing.opentelemetry import OpenTelemetryTracer

            return OpenTelemetryTracer(opentelemetry.trace.get_tracer("haystack"))
    except ImportError:
        pass

    return None


auto_enable_tracing()
