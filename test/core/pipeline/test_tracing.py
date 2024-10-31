# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from test.tracing.utils import SpyingSpan, SpyingTracer
from typing import Optional
from unittest.mock import ANY

import pytest
from _pytest.monkeypatch import MonkeyPatch

from haystack import Pipeline, component
from haystack.tracing.tracer import tracer


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: Optional[str]):  # use optional to spice up the typing tags
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
            pipeline_span := SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.input_data": {"hello": {"word": "world"}},
                    "haystack.pipeline.output_data": {"hello2": {"output": "Hello, Hello, world!!"}},
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_runs_per_component": 100,
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
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": []}},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": ["hello2"]}},
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
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": ["hello"]}},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": []}},
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
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": []}},
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
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": ["hello"]}},
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
