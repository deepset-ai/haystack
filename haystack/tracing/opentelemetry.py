# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import warnings
from collections.abc import Iterator
from typing import Any

from haystack.lazy_imports import LazyImport
from haystack.tracing import Span, Tracer
from haystack.tracing import utils as tracing_utils

with LazyImport("Run 'pip install opentelemetry-sdk'") as opentelemetry_import:
    import opentelemetry
    import opentelemetry.trace


class OpenTelemetrySpan(Span):
    def __init__(self, span: "opentelemetry.trace.Span") -> None:
        """Creates an instance of OpenTelemetrySpan."""
        self._span = span

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a single tag on the span.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        coerced_value = tracing_utils.coerce_tag_value(value)
        self._span.set_attribute(key, coerced_value)

    def raw_span(self) -> Any:
        """
        Provides access to the underlying span object of the tracer.

        :return: The underlying span object.
        """
        return self._span

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Return a dictionary with correlation data for logs."""
        span_context = self._span.get_span_context()
        return {"trace_id": span_context.trace_id, "span_id": span_context.span_id}


class OpenTelemetryTracer(Tracer):
    def __init__(self, tracer: "opentelemetry.trace.Tracer") -> None:
        """Creates an instance of OpenTelemetryTracer."""
        warnings.warn(
            "`OpenTelemetryTracer` will be removed from Haystack in version 3.0, as it is moving to "
            "the `opentelemetry-haystack` package. To continue using it, install that package with "
            "`pip install opentelemetry-haystack` and either add the `OpenTelemetryConnector` component to your "
            "pipeline or update your import to "
            "`from haystack_integrations.tracing.opentelemetry import OpenTelemetryTracer`. Note that OpenTelemetry "
            "tracing in Haystack 3.0 is no longer auto-enabled when `opentelemetry-sdk` is installed; use the "
            "`OpenTelemetryConnector` to enable it.",
            FutureWarning,
            stacklevel=2,
        )
        opentelemetry_import.check()
        self._tracer = tracer

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,  # noqa: ARG002
    ) -> Iterator[Span]:
        """Activate and return a new span that inherits from the current active span."""
        with self._tracer.start_as_current_span(operation_name) as raw_span:
            span = OpenTelemetrySpan(raw_span)
            if tags:
                span.set_tags(tags)

            yield span

    def current_span(self) -> Span | None:
        """Return the current active span"""
        current_span = opentelemetry.trace.get_current_span()
        if isinstance(current_span, opentelemetry.trace.NonRecordingSpan):
            return None

        return OpenTelemetrySpan(current_span)
