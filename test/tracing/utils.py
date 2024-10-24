# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import contextlib
import dataclasses
import uuid
from typing import Dict, Any, Optional, List, Iterator

from haystack.tracing import Span, Tracer


@dataclasses.dataclass
class SpyingSpan(Span):
    operation_name: str
    parent_span: Optional[Span] = None
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    trace_id: Optional[str] = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    span_id: Optional[str] = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "span_id": self.span_id}


class SpyingTracer(Tracer):
    def current_span(self) -> Optional[Span]:
        return self.spans[-1] if self.spans else None

    def __init__(self) -> None:
        self.spans: List[SpyingSpan] = []

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        new_span = SpyingSpan(operation_name, parent_span)

        for key, value in (tags or {}).items():
            new_span.set_tag(key, value)

        self.spans.append(new_span)

        yield new_span
