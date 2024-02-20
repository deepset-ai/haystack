import contextlib
import dataclasses
from typing import Dict, Any, List, Optional, Iterator

import pytest

from haystack import component, Pipeline
from haystack.tracing import Tracer, Span
from haystack.tracing.tracer import enable_tracing


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: str):
        """
        Takes a string in input and returns "Hello, <string>!"
        in output.
        """
        return {"output": f"Hello, {word}!"}


@dataclasses.dataclass
class SpyingSpan(Span):
    operation_name: str
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value


class SpyingTracer(Tracer):
    def current_span(self) -> Optional[Span]:
        return self.spans[-1] if self.spans else None

    def __init__(self) -> None:
        self.spans: List[Span] = []

    @contextlib.contextmanager
    def trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> Iterator[Span]:
        new_span = SpyingSpan(operation_name, tags)

        self.spans.append(new_span)

        yield new_span


@pytest.fixture()
def pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())
    pipeline.connect("hello.output", "hello2.word")
    return pipeline


class TestTracing:
    def test_with_enabled_tracing(self, pipeline: Pipeline) -> None:
        spying_tracer = SpyingTracer()
        enable_tracing(spying_tracer)

        pipeline.run(data={"word": "world"})

        assert len(spying_tracer.spans) == 3

        assert spying_tracer.spans == [
            SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.debug": False,
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_loops_allowed": 100,
                },
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "Hello",
                    "haystack.component.inputs": {"word": "str"},
                    "haystack.component.outputs": {"output": "str"},
                    "haystack.component.visits": 1,
                },
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.inputs": {"word": "str"},
                    "haystack.component.outputs": {"output": "str"},
                    "haystack.component.visits": 1,
                },
            ),
        ]
