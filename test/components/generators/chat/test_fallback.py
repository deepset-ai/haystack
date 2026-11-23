# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.generators.chat.fallback import FallbackChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT, StreamingChunk
from haystack.tools import ToolsType


@component
class _DummySuccessGen:
    def __init__(self, text: str = "ok"):
        self.text = text
        self.received_messages: list[list[ChatMessage]] = []

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, text=self.text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummySuccessGen":
        return default_from_dict(cls, data)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        self.received_messages.append(messages)
        if streaming_callback:
            streaming_callback(StreamingChunk(content=self.text))
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        self.received_messages.append(messages)
        if streaming_callback:
            streaming_callback(StreamingChunk(content=self.text))
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}


@component
class _DummyFailGen:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc or RuntimeError("boom")

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        raise self.exc

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        raise self.exc


@component
class CustomGeneratorWithoutSerDe:
    def __init__(self, text: str = "custom_ok"):
        self.text = text

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {}}


class TestInitSerde:
    def test_init(self):
        gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(), _DummyFailGen()])
        assert [type(generator) for generator in gen.chat_generators] == [_DummySuccessGen, _DummyFailGen]

    def test_init_with_empty_generators(self):
        with pytest.raises(ValueError, match="'chat_generators' must be a non-empty list"):
            FallbackChatGenerator(chat_generators=[])

    def test_serialization_roundtrip(self):
        original = FallbackChatGenerator(
            chat_generators=[_DummySuccessGen(text="hello"), CustomGeneratorWithoutSerDe(text="world")]
        )
        restored = FallbackChatGenerator.from_dict(original.to_dict())
        first, second = restored.chat_generators
        assert isinstance(first, _DummySuccessGen)
        assert first.text == "hello"
        assert isinstance(second, CustomGeneratorWithoutSerDe)
        assert second.text == "world"


class TestRun:
    def test_run_with_string_input(self):
        inner = _DummySuccessGen()
        gen = FallbackChatGenerator(chat_generators=[inner])
        gen.run("hi")
        assert inner.received_messages == [[ChatMessage.from_user("hi")]]

    def test_sequential_first_success(self):
        second = _DummySuccessGen(text="B")
        gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A"), second])
        result = gen.run([ChatMessage.from_user("hi")])
        assert result == {
            "replies": [ChatMessage.from_assistant("A")],
            "meta": {
                "dummy_meta": True,
                "successful_chat_generator_index": 0,
                "successful_chat_generator_class": "_DummySuccessGen",
                "total_attempts": 1,
                "failed_chat_generators": [],
            },
        }
        assert second.received_messages == []

    @pytest.mark.parametrize("failure_count", [1, 2])
    def test_succeeds_after_failures(self, failure_count):
        failing = [_DummyFailGen() for _ in range(failure_count)]
        gen = FallbackChatGenerator(chat_generators=[*failing, _DummySuccessGen(text="B")])
        result = gen.run([ChatMessage.from_user("hi")])
        assert result == {
            "replies": [ChatMessage.from_assistant("B")],
            "meta": {
                "dummy_meta": True,
                "successful_chat_generator_index": failure_count,
                "successful_chat_generator_class": "_DummySuccessGen",
                "total_attempts": failure_count + 1,
                "failed_chat_generators": ["_DummyFailGen"] * failure_count,
            },
        }

    def test_all_fail_raises(self):
        gen = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummyFailGen(exc=ValueError("last"))])
        with pytest.raises(
            RuntimeError,
            match=r"All 2 chat generators failed\. Last error: last\. "
            r"Failed chat generators: \[_DummyFailGen, _DummyFailGen\]",
        ):
            gen.run([ChatMessage.from_user("hi")])

    def test_streaming_callback_forwarding_sync(self):
        callback = Mock()
        gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen()])
        gen.run([ChatMessage.from_user("hi")], streaming_callback=callback)
        callback.assert_called_once_with(StreamingChunk(content="ok"))


class TestRunAsync:
    async def test_run_with_string_input(self):
        inner = _DummySuccessGen()
        gen = FallbackChatGenerator(chat_generators=[inner])
        await gen.run_async("hi")
        assert inner.received_messages == [[ChatMessage.from_user("hi")]]

    async def test_sequential_first_success(self):
        second = _DummySuccessGen(text="B")
        gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen(text="A"), second])
        result = await gen.run_async([ChatMessage.from_user("hi")])
        assert result == {
            "replies": [ChatMessage.from_assistant("A")],
            "meta": {
                "dummy_meta": True,
                "successful_chat_generator_index": 0,
                "successful_chat_generator_class": "_DummySuccessGen",
                "total_attempts": 1,
                "failed_chat_generators": [],
            },
        }
        assert second.received_messages == []

    @pytest.mark.parametrize("failure_count", [1, 2])
    async def test_succeeds_after_failures(self, failure_count):
        failing = [_DummyFailGen() for _ in range(failure_count)]
        gen = FallbackChatGenerator(chat_generators=[*failing, _DummySuccessGen(text="B")])
        result = await gen.run_async([ChatMessage.from_user("hi")])
        assert result == {
            "replies": [ChatMessage.from_assistant("B")],
            "meta": {
                "dummy_meta": True,
                "successful_chat_generator_index": failure_count,
                "successful_chat_generator_class": "_DummySuccessGen",
                "total_attempts": failure_count + 1,
                "failed_chat_generators": ["_DummyFailGen"] * failure_count,
            },
        }

    async def test_all_fail_raises(self):
        gen = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummyFailGen(exc=ValueError("last"))])
        with pytest.raises(
            RuntimeError,
            match=r"All 2 chat generators failed\. Last error: last\. "
            r"Failed chat generators: \[_DummyFailGen, _DummyFailGen\]",
        ):
            await gen.run_async([ChatMessage.from_user("hi")])

    async def test_streaming_callback_forwarding_async(self):
        callback = Mock()
        gen = FallbackChatGenerator(chat_generators=[_DummySuccessGen()])
        await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=callback)
        callback.assert_called_once_with(StreamingChunk(content="ok"))


class TestTracing:
    def test_pipeline_traces_output_only_on_fallback_component_span(self, spying_tracer):
        messages = [ChatMessage.from_user("hi")]
        fallback = FallbackChatGenerator(chat_generators=[_DummySuccessGen()])
        pipeline = Pipeline()
        pipeline.add_component(name="fallback", instance=fallback)

        pipeline.run(data={"fallback": {"messages": messages}})

        fallback_spans = [
            span
            for span in spying_tracer.spans
            if span.operation_name == "haystack.component.run"
            and span.tags["haystack.component.type"] == "FallbackChatGenerator"
        ]
        generator_spans = [span for span in spying_tracer.spans if span.operation_name == "haystack.chat_generator.run"]

        assert len(fallback_spans) == 1
        assert fallback_spans[0].tags["haystack.component.output"]["replies"][0].text == "ok"
        assert len(generator_spans) == 1
        assert "haystack.component.output" not in generator_spans[0].tags

    def test_run_traces_each_chat_generator_attempt(self, spying_tracer):
        messages = [ChatMessage.from_user("hi")]
        generation_kwargs = {"temperature": 0.1}
        fallback = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummySuccessGen()])

        with spying_tracer.trace("parent") as parent_span:
            fallback.run(messages=messages, generation_kwargs=generation_kwargs)

        generator_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.chat_generator.run"]
        assert len(generator_spans) == 2
        assert all(span.parent_span is parent_span for span in generator_spans)
        assert [span.tags["haystack.component.type"] for span in generator_spans] == [
            "_DummyFailGen",
            "_DummySuccessGen",
        ]
        assert all(
            span.tags["haystack.component.input"]
            == {"messages": messages, "generation_kwargs": generation_kwargs, "tools": None, "streaming_callback": None}
            for span in generator_spans
        )
        assert all("haystack.component.output" not in span.tags for span in generator_spans)

    @pytest.mark.asyncio
    async def test_run_async_traces_each_chat_generator_attempt(self, spying_tracer):
        messages = [ChatMessage.from_user("hi")]
        generation_kwargs = {"temperature": 0.1}
        fallback = FallbackChatGenerator(chat_generators=[_DummyFailGen(), _DummySuccessGen()])

        with spying_tracer.trace("parent") as parent_span:
            await fallback.run_async(messages=messages, generation_kwargs=generation_kwargs)

        generator_spans = [s for s in spying_tracer.spans if s.operation_name == "haystack.chat_generator.run"]
        assert len(generator_spans) == 2
        assert all(span.parent_span is parent_span for span in generator_spans)
        assert [span.tags["haystack.component.type"] for span in generator_spans] == [
            "_DummyFailGen",
            "_DummySuccessGen",
        ]
        assert all(
            span.tags["haystack.component.input"]
            == {"messages": messages, "generation_kwargs": generation_kwargs, "tools": None, "streaming_callback": None}
            for span in generator_spans
        )
        assert all("haystack.component.output" not in span.tags for span in generator_spans)


class TestComponentLifecycle:
    def test_warm_up_delegates_to_every_generator(self) -> None:

        gens = [Mock(spec=["run", "warm_up"]) for _ in range(3)]
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        fallback.warm_up()
        for gen in gens:
            gen.warm_up.assert_called_once()

    async def test_warm_up_async_delegates_to_every_generator(self) -> None:

        gens = [Mock(spec=["run", "warm_up_async"]) for _ in range(3)]
        for gen in gens:
            gen.warm_up_async = AsyncMock()
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        await fallback.warm_up_async()
        for gen in gens:
            gen.warm_up_async.assert_awaited_once()

    async def test_warm_up_async_falls_back_to_sync_warm_up(self) -> None:

        gens = [Mock(spec=["run", "warm_up"]) for _ in range(3)]
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        await fallback.warm_up_async()
        for gen in gens:
            gen.warm_up.assert_called_once()

    def test_close_delegates_to_every_generator(self) -> None:

        gens = [Mock(spec=["run", "close"]) for _ in range(3)]
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        fallback.close()
        for gen in gens:
            gen.close.assert_called_once()

    async def test_close_async_delegates_to_every_generator(self) -> None:

        gens = [Mock(spec=["run", "close_async"]) for _ in range(3)]
        for gen in gens:
            gen.close_async = AsyncMock()
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        await fallback.close_async()
        for gen in gens:
            gen.close_async.assert_awaited_once()

    async def test_close_async_falls_back_to_sync_close(self) -> None:

        gens = [Mock(spec=["run", "close"]) for _ in range(3)]
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        await fallback.close_async()
        for gen in gens:
            gen.close.assert_called_once()

    async def test_lifecycle_is_safe_when_generators_lack_methods(self) -> None:

        gens = [Mock(spec=["run"]) for _ in range(3)]
        fallback = FallbackChatGenerator(chat_generators=gens)  # type: ignore[arg-type]
        fallback.warm_up()
        fallback.close()
        await fallback.close_async()
