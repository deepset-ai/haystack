# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
from typing import Any, Dict, Iterator, Optional

from haystack import logging
from haystack.tracing import Span, Tracer
from haystack.tracing.utils import coerce_tag_value

logger = logging.getLogger(__name__)

RESET_COLOR = "\033[0m"


@dataclasses.dataclass
class LoggingSpan(Span):
    operation_name: str
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a single tag on the span.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        self.tags[key] = value


class LoggingTracer(Tracer):
    """
    A simple tracer that logs the operation name and tags of a span.
    """

    def __init__(self, tags_color_strings: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the LoggingTracer.

        :param tags_color_strings:
            A dictionary that maps tag names to color strings that should be used when logging the tags.
            The color strings should be in the format of
            [ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code#Colors).
            For example, to color the tag "haystack.component.input" in red, you would pass
            `tags_color_strings={"haystack.component.input": "\x1b[1;31m"}`.
        """

        self.tags_color_strings = tags_color_strings or {}

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        """
        Trace the execution of a block of code.

        :param operation_name: the name of the operation being traced.
        :param tags: tags to apply to the newly created span.
        :param parent_span: the parent span to use for the newly created span. Not used in this simple tracer.
        :returns: the newly created span.
        """

        custom_span = LoggingSpan(operation_name, tags=tags or {})

        try:
            yield custom_span
        except Exception as e:
            raise e
        # we make sure to log the operation name and tags of the span when the context manager exits
        # both in case of success and error
        finally:
            operation_name = custom_span.operation_name
            tags = custom_span.tags or {}
            logger.debug("Operation: {operation_name}", operation_name=operation_name)
            for tag_name, tag_value in tags.items():
                color_string = self.tags_color_strings.get(tag_name, "")
                coerced_value = coerce_tag_value(tag_value)
                logger.debug(
                    color_string + "{tag_name}={tag_value}" + RESET_COLOR, tag_name=tag_name, tag_value=coerced_value
                )

    def current_span(self) -> Optional[Span]:
        """Return the current active span, if any."""
        # we don't store spans in this simple tracer
        return None
