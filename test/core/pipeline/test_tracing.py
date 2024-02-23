import pytest
from _pytest.monkeypatch import MonkeyPatch

from haystack import component, Pipeline
from haystack.tracing.tracer import enable_tracing, tracer
from test.tracing.utils import SpyingSpan, SpyingTracer


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: str):
        """
        Takes a string in input and returns "Hello, <string>!"
        in output.
        """
        return {"output": f"Hello, {word}!"}


@pytest.fixture()
def pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())
    pipeline.connect("hello.output", "hello2.word")
    return pipeline


class TestTracing:
    def test_with_enabled_tracing(self, pipeline: Pipeline, spying_tracer: SpyingTracer) -> None:
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
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str", "senders": []}},
                    "haystack.component.output_spec": {"output": {"type": "str", "senders": ["hello2"]}},
                    "haystack.component.visits": 1,
                },
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str", "senders": ["hello"]}},
                    "haystack.component.output_spec": {"output": {"type": "str", "senders": []}},
                    "haystack.component.visits": 1,
                },
            ),
        ]

    def test_with_enabled_content_tracing(
        self, spying_tracer: SpyingTracer, monkeypatch: MonkeyPatch, pipeline: Pipeline
    ) -> None:
        # Monkeypatch to avoid impact on other tests
        monkeypatch.setattr(tracer, "is_content_tracing_enabled", True)

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
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str", "senders": []}},
                    "haystack.component.output_spec": {"output": {"type": "str", "senders": ["hello2"]}},
                    "haystack.component.input": {"word": "world"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, world!"},
                },
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str", "senders": ["hello"]}},
                    "haystack.component.output_spec": {"output": {"type": "str", "senders": []}},
                    "haystack.component.input": {"word": "Hello, world!"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, Hello, world!!"},
                },
            ),
        ]
