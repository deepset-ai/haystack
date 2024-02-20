import abc
import contextlib
from typing import Dict, Any, Optional, Iterator


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


class Tracer(abc.ABC):
    """Interface for instrumenting code by creating and submitting spans."""

    @abc.abstractmethod
    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]]) -> Iterator[Span]:
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

    def current_raw_span(self) -> Optional[Any]:
        """Provides access to the underlying span object of the tracer.

        Use this if you need full access to the underlying span object.

        :return: The underlying span object or `None` if no span is active.
        """
        # By default, we just return the span object, but this can be overridden by subclasses if they need to return
        return self.current_span()


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


_tracer: Tracer = NullTracer()


# We use a method to access the global tracer instance so that we can conveniently swap out the tracer
# (if folks would import the object directly, we'd have to monkey-patch it in all of these modules).
def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return _tracer


def enable_tracing(provided_tracer: Tracer) -> None:
    """Enable tracing by setting the global tracer instance."""
    global _tracer  # pylint: disable=global-statement
    _tracer = provided_tracer


def disable_tracing() -> None:
    """Disable tracing by setting the global tracer instance to a no-op tracer."""
    global _tracer  # pylint: disable=global-statement
    _tracer = NullTracer()


def is_tracing_enabled() -> bool:
    """Return whether tracing is enabled."""
    return not isinstance(_tracer, NullTracer)
