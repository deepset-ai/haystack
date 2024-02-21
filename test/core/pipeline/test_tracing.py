import pytest

from haystack import component, Pipeline
from haystack.tracing.tracer import enable_tracing
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
