import contextlib
from typing import Any, Dict, Iterator, Optional

from haystack.tracing import Span, Tracer
import dataclasses
from typing import Dict, Any, Optional, Iterator

from haystack import logging


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LoggingSpan(Span):
    operation_name: str
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


print(Tracer)


class LoggingTracer(Tracer):
    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        """Activate and return a new span that inherits from the current active span."""

        custom_span = LoggingSpan(operation_name, tags=tags or {})

        try:
            yield custom_span
        except Exception as e:
            raise e
        finally:
            logger.debug(
                "Operation: {operation_name} - Tags: {tags}",
                operation_name=custom_span.operation_name,
                tags=custom_span.tags,
            )

    def current_span(self) -> Optional[Span]:
        return None
