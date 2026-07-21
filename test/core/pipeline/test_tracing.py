# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace
from unittest.mock import ANY

import pytest
from _pytest.monkeypatch import MonkeyPatch

from haystack import Pipeline, component
from haystack.dataclasses import Document
from haystack.tracing.tracer import disable_tracing, enable_tracing, tracer
from test.tracing.utils import EagerSpyingTracer, SpyingSpan, SpyingTracer


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: str | None) -> dict[str, str]:  # use optional to spice up the typing tags
        """
        Takes a string in input and returns "Hello, <string>!" in output.
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
            pipeline_span := SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.input_data": {"hello": {"word": "world"}},
                    "haystack.pipeline.output_data": {"hello2": {"output": "Hello, Hello, world!!"}},
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_runs_per_component": 100,
                    "haystack.pipeline.execution_mode": "sync",
                },
                parent_span=None,
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "Hello",
                    "haystack.component.fully_qualified_type": "test.core.pipeline.test_tracing.Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": []}},
                    "haystack.component.input": {"word": "world"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": ["hello2"]}},
                    "haystack.component.output": {"output": "Hello, world!"},
                    "haystack.component.visits": 1,
                },
                parent_span=pipeline_span,
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.fully_qualified_type": "test.core.pipeline.test_tracing.Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": ["hello"]}},
                    "haystack.component.input": {"word": "Hello, world!"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": []}},
                    "haystack.component.output": {"output": "Hello, Hello, world!!"},
                    "haystack.component.visits": 1,
                },
                parent_span=pipeline_span,
                trace_id=ANY,
                span_id=ANY,
            ),
        ]

        # We need to check the type of the input_spec because it can be rendered differently
        # depending on the Python version 🫠
        assert spying_tracer.spans[1].tags["haystack.component.input_spec"]["word"]["type"] in [
            "typing.Union[str, NoneType]",
            "typing.Optional[str]",
            "str | None",
        ]

    def test_with_enabled_content_tracing(
        self, spying_tracer: SpyingTracer, monkeypatch: MonkeyPatch, pipeline: Pipeline
    ) -> None:
        # Monkeypatch to avoid impact on other tests
        monkeypatch.setattr(tracer, "is_content_tracing_enabled", True)

        pipeline.run(data={"word": "world"})

        assert len(spying_tracer.spans) == 3
        assert spying_tracer.spans == [
            pipeline_span := SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_runs_per_component": 100,
                    "haystack.pipeline.input_data": {"hello": {"word": "world"}},
                    "haystack.pipeline.output_data": {"hello2": {"output": "Hello, Hello, world!!"}},
                    "haystack.pipeline.execution_mode": "sync",
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "Hello",
                    "haystack.component.fully_qualified_type": "test.core.pipeline.test_tracing.Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str | None", "senders": []}},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": ["hello2"]}},
                    "haystack.component.input": {"word": "world"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, world!"},
                },
                parent_span=pipeline_span,
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.fully_qualified_type": "test.core.pipeline.test_tracing.Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": "str | None", "senders": ["hello"]}},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": []}},
                    "haystack.component.input": {"word": "Hello, world!"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, Hello, world!!"},
                },
                parent_span=pipeline_span,
                trace_id=ANY,
                span_id=ANY,
            ),
        ]

    @pytest.mark.parametrize("run_async", [False, True])
    def test_pipeline_output_data_recorded_after_run(self, pipeline, run_async, monkeypatch):
        """
        Verify the haystack.pipeline.output_data tag holds the final pipeline outputs.

        Real tracing backends coerce a tag value when `set_tag` is called (at span start), not lazily at
        span end. If output_data is set from the still-empty outputs dict at span start, production traces
        capture an empty pipeline output. Parametrized to cover the sync (`run`) and async (`run_async`) paths.
        """
        monkeypatch.setattr(tracer, "is_content_tracing_enabled", True)
        eager_tracer = EagerSpyingTracer()
        enable_tracing(eager_tracer)
        try:
            if run_async:
                asyncio.run(pipeline.run_async(data={"word": "world"}))
            else:
                pipeline.run(data={"word": "world"})
        finally:
            disable_tracing()

        pipeline_span = eager_tracer.spans[0]
        assert pipeline_span.operation_name == "haystack.pipeline.run"
        assert json.loads(pipeline_span.tags["haystack.pipeline.input_data"]) == {"hello": {"word": "world"}}
        assert json.loads(pipeline_span.tags["haystack.pipeline.output_data"]) == {
            "hello2": {"output": "Hello, Hello, world!!"}
        }

    @pytest.mark.parametrize("run_async", [False, True])
    def test_pipeline_output_data_gated_by_content_tracing(self, pipeline, run_async, monkeypatch):
        """
        output_data carries the pipeline's answer content, so it is emitted only when content tracing is
        enabled. input_data stays a normal tag and is always emitted.
        """
        monkeypatch.setattr(tracer, "is_content_tracing_enabled", False)
        eager_tracer = EagerSpyingTracer()
        enable_tracing(eager_tracer)
        try:
            if run_async:
                asyncio.run(pipeline.run_async(data={"word": "world"}))
            else:
                pipeline.run(data={"word": "world"})
        finally:
            disable_tracing()

        pipeline_span = eager_tracer.spans[0]
        assert "haystack.pipeline.output_data" not in pipeline_span.tags
        assert json.loads(pipeline_span.tags["haystack.pipeline.input_data"]) == {"hello": {"word": "world"}}
        assert pipeline_span.tags["haystack.pipeline.execution_mode"] == ("async" if run_async else "sync")

    @pytest.mark.parametrize("run_async", [False, True])
    def test_span_input_not_affected_by_component_mutation(self, run_async, spying_tracer, monkeypatch):
        """
        Verify that the haystack.component.input span tag retains the pre-execution value.

        Parametrized to cover both the synchronous (`run`) and asynchronous (`run_async`) execution paths.
        """
        monkeypatch.setattr(tracer, "is_content_tracing_enabled", True)

        @component
        class MutatingComponent:
            @component.output_types(doc=Document)
            def run(self, doc: Document) -> dict:
                return {"doc": replace(doc, content="mutated")}

        pipe = Pipeline()
        pipe.add_component("mutator", MutatingComponent())

        if run_async:
            result = asyncio.run(pipe.run_async({"mutator": {"doc": Document(content="original")}}))
        else:
            result = pipe.run({"mutator": {"doc": Document(content="original")}})

        pipeline_span = spying_tracer.spans[0]
        component_span = spying_tracer.spans[1]

        assert pipeline_span.operation_name == "haystack.pipeline.run"
        assert pipeline_span.tags["haystack.pipeline.execution_mode"] == ("async" if run_async else "sync")
        assert component_span.tags["haystack.component.input"]["doc"].content == "original"
        assert result["mutator"]["doc"].content == "mutated"
