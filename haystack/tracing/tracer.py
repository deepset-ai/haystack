# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import abc
import contextlib
import os
from collections.abc import Iterator
from typing import Any

HAYSTACK_CONTENT_TRACING_ENABLED_ENV_VAR = "HAYSTACK_CONTENT_TRACING_ENABLED"


class Span(abc.ABC):
    """Interface for an instrumented operation."""

    @abc.abstractmethod
    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a single tag on the span.

        Note that the value will be serialized to a string, so it's best to use simple types like strings, numbers, or
        booleans.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        pass

    def set_tags(self, tags: dict[str, Any]) -> None:
        """
        Set multiple tags on the span.

        :param tags: a mapping of tag names to tag values.
        """
        for key, value in tags.items():
            self.set_tag(key, value)

    def raw_span(self) -> Any:
        """
        Provides access to the underlying span object of the tracer.

        Use this if you need full access to the underlying span object.

        :return: The underlying span object.
        """
        return self

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Set a single tag containing content information.

        Content is sensitive information such as
        - the content of a query
        - the content of a document
        - the content of an answer

        By default, this behavior is disabled. To enable it
        - set the environment variable `HAYSTACK_CONTENT_TRACING_ENABLED` to `true` or
        - override the `set_content_tag` method in a custom tracer implementation.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        if tracer.is_content_tracing_enabled:
            self.set_tag(key, value)

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """
        Return a dictionary with correlation data for logs.

        This is useful if you want to correlate logs with traces.
        """
        return {}


class Tracer(abc.ABC):
    """Interface for instrumenting code by creating and submitting spans."""

    @abc.abstractmethod
    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """
        Trace the execution of a block of code.

        :param operation_name: the name of the operation being traced.
        :param tags: tags to apply to the newly created span.
        :param parent_span: the parent span to use for the newly created span.
            If `None`, the newly created span will be a root span.
        :return: the newly created span.
        """
        pass

    @abc.abstractmethod
    def current_span(self) -> Span | None:
        """
        Returns the currently active span. If no span is active, returns `None`.

        :return: Currently active span or `None` if no span is active.
        """
        pass


class ProxyTracer(Tracer):
    """
    Container for the actual tracer instance.

    This eases
    - replacing the actual tracer instance without having to change the global tracer instance
    - implementing default behavior for the tracer
    """

    def __init__(self, provided_tracer: Tracer) -> None:
        """Creates an instance of ProxyTracer."""
        self.actual_tracer: Tracer = provided_tracer
        self.is_content_tracing_enabled = os.getenv(HAYSTACK_CONTENT_TRACING_ENABLED_ENV_VAR, "false").lower() == "true"

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Activate and return a new span that inherits from the current active span."""
        with self.actual_tracer.trace(operation_name, tags=tags, parent_span=parent_span) as span:
            yield span

    def current_span(self) -> Span | None:
        """Return the current active span"""
        return self.actual_tracer.current_span()


class NullSpan(Span):
    """A no-op implementation of the `Span` interface. This is used when tracing is disabled."""

    def set_tag(self, key: str, value: Any) -> None:
        """Set a single tag on the span."""
        pass


class NullTracer(Tracer):
    """A no-op implementation of the `Tracer` interface. This is used when tracing is disabled."""

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,  # noqa: ARG002
        tags: dict[str, Any] | None = None,  # noqa: ARG002
        parent_span: Span | None = None,  # noqa: ARG002
    ) -> Iterator[Span]:
        """Activate and return a new span that inherits from the current active span."""
        yield NullSpan()

    def current_span(self) -> Span | None:
        """Return the current active span"""
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
