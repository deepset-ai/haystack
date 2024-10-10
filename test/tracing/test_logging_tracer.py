# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import functools
import json
from typing import List, Dict, Optional
import logging


import pytest

from haystack.tracing.logging_tracer import LoggingTracer
from haystack import component, Pipeline
from haystack import tracing


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: Optional[str]):
        return {"output": f"Hello, {word}!"}


class TestLoggingTracer:
    def test_init(self) -> None:
        tracer = LoggingTracer()
        assert tracer

    def test_logging_tracer(self, caplog) -> None:
        tracer = LoggingTracer()

        caplog.set_level(logging.DEBUG)
        with tracer.trace("test") as span:
            span.set_tag("key", "value")

        assert "Operation: test" in caplog.text
        assert "key=value" in caplog.text
        assert len(caplog.records) == 2

        # structured logging
        assert caplog.records[0].operation_name == "test"
        assert caplog.records[1].tag_name == "key"
        assert caplog.records[1].tag_value == "value"

    def test_tracing_complex_values(self, caplog) -> None:
        tracer = LoggingTracer()

        caplog.set_level(logging.DEBUG)

        with tracer.trace("test") as span:
            span.set_tag("key", {"a": 1, "b": [2, 3, 4]})

        assert "Operation: test" in caplog.text
        assert "key={'a': 1, 'b': [2, 3, 4]}" in caplog.text
        assert len(caplog.records) == 2

        # structured logging
        assert caplog.records[0].operation_name == "test"
        assert caplog.records[1].tag_name == "key"
        assert caplog.records[1].tag_value == {"a": 1, "b": [2, 3, 4]}

    def test_logging_pipeline(self, caplog) -> None:
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())
        pipeline.add_component("hello2", Hello())
        pipeline.connect("hello.output", "hello2.word")

        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        pipeline.run(data={"word": "world"})

        records = caplog.records

        assert any(
            record.operation_name == "haystack.component.run" for record in records if hasattr(record, "operation_name")
        )
        assert any(
            record.operation_name == "haystack.pipeline.run" for record in records if hasattr(record, "operation_name")
        )

        tags_records = [record for record in records if hasattr(record, "tag_name")]
        assert any(record.tag_name == "haystack.component.name" for record in tags_records)
        assert any(record.tag_value == "hello" for record in tags_records)
