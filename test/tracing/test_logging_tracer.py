# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import logging


from haystack.tracing.logging_tracer import LoggingTracer
from haystack import component, Pipeline
from haystack import tracing


@component
class Hello:
    @component.output_types(output=str)
    def run(self, word: Optional[str]):
        return {"output": f"Hello, {word}!"}


@component
class FailingComponent:
    @component.output_types(output=str)
    def run(self, word: Optional[str]):
        raise Exception("Failing component")


class TestLoggingTracer:
    def test_init(self) -> None:
        tracer = LoggingTracer()
        assert tracer.tags_color_strings == {}

        tracer = LoggingTracer(tags_color_strings={"tag_name": "color_string"})
        assert tracer.tags_color_strings == {"tag_name": "color_string"}

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

    def test_apply_color_strings(self, caplog) -> None:
        tracer = LoggingTracer(tags_color_strings={"key": "color_string"})

        caplog.set_level(logging.DEBUG)

        with tracer.trace("test") as span:
            span.set_tag("key", "value")

        assert "color_string" in caplog.text

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

        tracing.disable_tracing()

    def test_logging_pipeline_with_content_tracing(self, caplog) -> None:
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())

        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())

        caplog.set_level(logging.DEBUG)

        pipeline.run(data={"word": "world"})
        records = caplog.records

        tags_records = [record for record in records if hasattr(record, "tag_name")]

        input_tag_value = [
            record.tag_value for record in tags_records if record.tag_name == "haystack.component.input"
        ][0]
        assert input_tag_value == {"word": "world"}

        output_tag_value = [
            record.tag_value for record in tags_records if record.tag_name == "haystack.component.output"
        ][0]
        assert output_tag_value == {"output": "Hello, world!"}

        tracing.tracer.is_content_tracing_enabled = False
        tracing.disable_tracing()

    def test_logging_pipeline_on_failure(self, caplog) -> None:
        """
        Test that the LoggingTracer also logs events when a component fails.
        """
        pipeline = Pipeline()
        pipeline.add_component("failing_component", FailingComponent())

        tracing.enable_tracing(LoggingTracer())
        caplog.set_level(logging.DEBUG)

        try:
            pipeline.run(data={"word": "world"})
        except:
            pass

        records = caplog.records

        assert any(
            record.operation_name == "haystack.component.run" for record in records if hasattr(record, "operation_name")
        )
        assert any(
            record.operation_name == "haystack.pipeline.run" for record in records if hasattr(record, "operation_name")
        )

        tags_records = [record for record in records if hasattr(record, "tag_name")]
        assert any(record.tag_name == "haystack.component.name" for record in tags_records)
        assert any(record.tag_value == "failing_component" for record in tags_records)

        tracing.disable_tracing()
