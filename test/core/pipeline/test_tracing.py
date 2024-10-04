# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from test.tracing.utils import SpyingSpan, SpyingTracer
from typing import List, Optional
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


@component
class HelloList:
    @component.output_types(output=List[str])
    def run(self, words: Optional[List[str]]):  # use optional to spice up the typing tags
        """
        Takes a string in input and returns "Hello, <string>!"
        in output.
        """
        outputs = [f"Hello, {word}!" for word in words]
        return {"output": outputs}


@pytest.fixture()
def pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("hello", Hello())
    pipeline.add_component("hello2", Hello())
    pipeline.connect("hello.output", "hello2.word")
    return pipeline


@pytest.fixture()
def pipeline_list_input() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("hello", HelloList())
    return pipeline


class TestTracing:
    def test_with_enabled_tracing(self, pipeline: Pipeline, spying_tracer: SpyingTracer) -> None:
        pipeline.run(data={"word": "world"})

        assert len(spying_tracer.spans) == 3

        assert spying_tracer.spans == [
            SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.input_data": {"hello": {"word": "world"}},
                    "haystack.pipeline.output_data": {"hello2": {"output": "Hello, Hello, world!!"}},
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_loops_allowed": 100,
                    "haystack.pipeline.max_runs_per_component": 100,
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_lengths": {},
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": []}},
                    "haystack.component.output_lengths": {},
                    "haystack.component.output_types": {"output": "str"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": ["hello2"]}},
                    "haystack.component.visits": 1,
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_lengths": {},
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": ["hello"]}},
                    "haystack.component.output_lengths": {},
                    "haystack.component.output_types": {"output": "str"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": []}},
                    "haystack.component.visits": 1,
                },
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
            SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_loops_allowed": 100,
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
                    "haystack.component.input_lengths": {},
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": []}},
                    "haystack.component.output_lengths": {},
                    "haystack.component.output_types": {"output": "str"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": ["hello2"]}},
                    "haystack.component.input": {"word": "world"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, world!"},
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello2",
                    "haystack.component.type": "Hello",
                    "haystack.component.input_lengths": {},
                    "haystack.component.input_types": {"word": "str"},
                    "haystack.component.input_spec": {"word": {"type": ANY, "senders": ["hello"]}},
                    "haystack.component.output_lengths": {},
                    "haystack.component.output_types": {"output": "str"},
                    "haystack.component.output_spec": {"output": {"type": "str", "receivers": []}},
                    "haystack.component.input": {"word": "Hello, world!"},
                    "haystack.component.visits": 1,
                    "haystack.component.output": {"output": "Hello, Hello, world!!"},
                },
                trace_id=ANY,
                span_id=ANY,
            ),
        ]

    def test_with_enabled_tracing_input_lengths(
        self, pipeline_list_input: Pipeline, spying_tracer: SpyingTracer
    ) -> None:
        pipeline_list_input.run(data={"words": ["happy", "world"]})

        assert len(spying_tracer.spans) == 2

        assert spying_tracer.spans == [
            SpyingSpan(
                operation_name="haystack.pipeline.run",
                tags={
                    "haystack.pipeline.input_data": {"hello": {"words": ["happy", "world"]}},
                    "haystack.pipeline.output_data": {"hello": {"output": ["Hello, happy!", "Hello, world!"]}},
                    "haystack.pipeline.metadata": {},
                    "haystack.pipeline.max_loops_allowed": 100,
                    "haystack.pipeline.max_runs_per_component": 100,
                },
                trace_id=ANY,
                span_id=ANY,
            ),
            SpyingSpan(
                operation_name="haystack.component.run",
                tags={
                    "haystack.component.name": "hello",
                    "haystack.component.type": "HelloList",
                    "haystack.component.input_lengths": {"words": 2},
                    "haystack.component.input_types": {"words": "list"},
                    "haystack.component.input_spec": {"words": {"type": ANY, "senders": []}},
                    "haystack.component.output_lengths": {"output": 2},
                    "haystack.component.output_types": {"output": "list"},
                    "haystack.component.output_spec": {"output": {"type": "typing.List[str]", "receivers": []}},
                    "haystack.component.visits": 1,
                },
                trace_id=ANY,
                span_id=ANY,
            ),
        ]

        # We need to check the type of the input_spec because it can be rendered differently
        # depending on the Python version 🫠
        assert spying_tracer.spans[1].tags["haystack.component.input_spec"]["words"]["type"] in [
            "typing.Union[typing.List[str], NoneType]",
            "typing.Optional[typing.List[str]]",
        ]
