# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging
from typing import Any

import pytest

from haystack import Pipeline, component
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import AsyncStreamingCallbackT, StreamingChunk


@component
class StreamingEcho:
    """Streaming component used by tests: emits `n_chunks` chunks then returns a final reply."""

    def __init__(
        self,
        prefix: str = "tok",
        n_chunks: int = 3,
        fail: bool = False,
        chunk_delay: float = 0.0,
        streaming_callback: AsyncStreamingCallbackT | None = None,
    ) -> None:
        self.prefix = prefix
        self.n_chunks = n_chunks
        self.fail = fail
        self.chunk_delay = chunk_delay
        self.streaming_callback = streaming_callback

    @component.output_types(reply=str)
    def run(self, prompt: str, streaming_callback: AsyncStreamingCallbackT | None = None) -> dict:
        return {"reply": f"{self.prefix}-final"}

    @component.output_types(reply=str)
    async def run_async(self, prompt: str, streaming_callback: AsyncStreamingCallbackT | None = None) -> dict:
        for i in range(self.n_chunks):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            chunk = StreamingChunk(content=f"{self.prefix}{i}")
            if streaming_callback is not None:
                await streaming_callback(chunk)
        if self.fail:
            raise RuntimeError("boom")
        return {"reply": f"{self.prefix}-final"}


@component
class DynamicStreamingEcho:
    """Streaming component that declares `streaming_callback` via `set_input_type` instead of the run signature."""

    def __init__(self, prefix: str = "tok", n_chunks: int = 2) -> None:
        self.prefix = prefix
        self.n_chunks = n_chunks
        self.streaming_callback = None
        component.set_input_type(self, "prompt", str)
        component.set_input_type(self, "streaming_callback", AsyncStreamingCallbackT | None, None)

    @component.output_types(reply=str)
    def run(self, **kwargs: Any) -> dict:
        return {"reply": f"{self.prefix}-final"}

    @component.output_types(reply=str)
    async def run_async(self, **kwargs: Any) -> dict:
        cb = kwargs.get("streaming_callback")
        for i in range(self.n_chunks):
            chunk = StreamingChunk(content=f"{self.prefix}{i}")
            if cb is not None:
                await cb(chunk)
        return {"reply": f"{self.prefix}-final"}


@component
class Passthrough:
    """Non-streaming async component used by filter-validation tests."""

    @component.output_types(prompt=str)
    def run(self, prompt: str) -> dict:
        return {"prompt": prompt}

    @component.output_types(prompt=str)
    async def run_async(self, prompt: str) -> dict:
        return {"prompt": prompt}


@pytest.mark.asyncio
async def test_stream_yields_chunks_and_returns_result():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=3))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi"}})
    chunks = [c async for c in handle]

    assert [c.content for c in chunks] == ["s0", "s1", "s2"]
    assert handle.result["streamer"] == {"reply": "s-final"}


@pytest.mark.asyncio
async def test_stream_yields_chunks_with_flat_input():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=3))

    # flat input form (`{"prompt": ...}` instead of `{"streamer": {"prompt": ...}}`)
    handle = pipeline.stream(data={"prompt": "hi"})
    chunks = [c async for c in handle]

    assert [c.content for c in chunks] == ["s0", "s1", "s2"]
    assert handle.result["streamer"] == {"reply": "s-final"}


@pytest.mark.asyncio
async def test_stream_composes_with_init_streaming_callback():
    seen = []

    async def init_callback(chunk: StreamingChunk) -> None:
        seen.append(chunk.content)

    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=2, streaming_callback=init_callback))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi"}})
    chunks = [c async for c in handle]

    assert [c.content for c in chunks] == ["s0", "s1"]
    assert seen == ["s0", "s1"]


@pytest.mark.asyncio
async def test_stream_runtime_callback_overrides_init():
    init_seen = []
    runtime_seen = []

    async def init_callback(chunk: StreamingChunk) -> None:
        init_seen.append(chunk.content)

    async def runtime_callback(chunk: StreamingChunk) -> None:
        runtime_seen.append(chunk.content)

    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=2, streaming_callback=init_callback))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi", "streaming_callback": runtime_callback}})
    [c async for c in handle]

    assert init_seen == []
    assert runtime_seen == ["s0", "s1"]


@pytest.mark.asyncio
async def test_stream_warns_on_sync_runtime_callback(caplog):
    seen: list[str] = []

    def sync_callback(chunk: StreamingChunk) -> None:
        seen.append(chunk.content)

    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=2))

    with caplog.at_level(logging.WARNING):
        handle = pipeline.stream(data={"streamer": {"prompt": "hi", "streaming_callback": sync_callback}})
    [c async for c in handle]

    assert "sync streaming callback" in caplog.text
    assert seen == ["s0", "s1"]


@pytest.mark.asyncio
async def test_stream_detects_streaming_callback_declared_via_set_input_type():
    pipeline = Pipeline()
    pipeline.add_component("streamer", DynamicStreamingEcho(prefix="d", n_chunks=3))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi"}})
    chunks = [c async for c in handle]

    assert [c.content for c in chunks] == ["d0", "d1", "d2"]
    assert handle.result["streamer"] == {"reply": "d-final"}


@pytest.mark.asyncio
async def test_stream_streams_all_components_by_default():
    pipeline = Pipeline()
    pipeline.add_component("a", StreamingEcho(prefix="a", n_chunks=2))
    pipeline.add_component("b", StreamingEcho(prefix="b", n_chunks=2))

    handle = pipeline.stream(data={"a": {"prompt": "x"}, "b": {"prompt": "y"}})
    chunks = [c async for c in handle]

    assert {c.content for c in chunks} == {"a0", "a1", "b0", "b1"}


@pytest.mark.asyncio
async def test_stream_filters_to_selected_components():
    pipeline = Pipeline()
    pipeline.add_component("a", StreamingEcho(prefix="a", n_chunks=2))
    pipeline.add_component("b", StreamingEcho(prefix="b", n_chunks=2))

    handle = pipeline.stream(data={"a": {"prompt": "x"}, "b": {"prompt": "y"}}, streaming_components=["a"])
    chunks = [c async for c in handle]

    assert {c.content for c in chunks} == {"a0", "a1"}
    assert handle.result["a"] == {"reply": "a-final"}
    assert handle.result["b"] == {"reply": "b-final"}


def test_stream_raises_for_unknown_component_in_filter():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho())

    with pytest.raises(ValueError, match="Unknown components") as excinfo:
        pipeline.stream(data={"streamer": {"prompt": "x"}}, streaming_components=["typo"])
    assert "typo" in str(excinfo.value)  # the message names the offending component


def test_stream_raises_for_non_streaming_component_in_filter():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho())
    pipeline.add_component("passthrough", Passthrough())

    with pytest.raises(ValueError, match="do not support streaming") as excinfo:
        pipeline.stream(
            data={"streamer": {"prompt": "x"}, "passthrough": {"prompt": "x"}}, streaming_components=["passthrough"]
        )
    assert "passthrough" in str(excinfo.value)  # the message names the offending component


@pytest.mark.asyncio
async def test_stream_propagates_pipeline_exception_during_iteration():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=1, fail=True))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    with pytest.raises(PipelineRuntimeError) as excinfo:
        async for _ in handle:
            pass

    # the error message indicates the failing component and the underlying error
    message = str(excinfo.value)
    assert "failed to run" in message
    assert "Component name: 'streamer'" in message
    assert "Component type: 'StreamingEcho'" in message
    assert "boom" in message

    # the original exception is preserved as the direct cause
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "boom"


@pytest.mark.asyncio
async def test_failing_callback_does_not_drop_chunk():
    async def failing_callback(chunk: StreamingChunk) -> None:
        raise RuntimeError("callback boom")

    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=3, streaming_callback=failing_callback))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi"}})

    received = []
    with pytest.raises(PipelineRuntimeError) as excinfo:
        async for chunk in handle:
            received.append(chunk.content)

    assert received == ["s0"]  # queued before the callback raised

    # the error message indicates the failing component and the underlying error
    message = str(excinfo.value)
    assert "failed to run" in message
    assert "Component name: 'streamer'" in message
    assert "callback boom" in message

    # the original exception is preserved as the direct cause
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "callback boom"


@pytest.mark.asyncio
async def test_result_reraises_original_failure():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=1, fail=True))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    with pytest.raises(PipelineRuntimeError) as iter_excinfo:
        async for _ in handle:
            pass

    # `result` re-raises the exception raised during iteration
    with pytest.raises(PipelineRuntimeError) as result_excinfo:
        handle.result
    assert result_excinfo.value is iter_excinfo.value


@pytest.mark.asyncio
async def test_result_raises_when_pipeline_not_finished():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=5))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    # no await between `stream()` and reading `result`, so the task has not started yet
    with pytest.raises(RuntimeError, match="Pipeline has not finished"):
        handle.result

    await handle.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_pipeline_and_result_reports_cancelled():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=100))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    await handle.aclose()

    with pytest.raises(RuntimeError, match="Pipeline was cancelled"):
        handle.result


@pytest.mark.asyncio
async def test_consumer_cancellation_cancels_pipeline():
    pipeline = Pipeline()
    # chunk_delay keeps the producer running so the consumer can be cancelled mid-stream
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=100, chunk_delay=0.01))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    async def consumer() -> None:
        async for _ in handle:
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.03)  # let the producer emit a couple of chunks
    # send a cancel request and wait for the consumer to finish
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert handle._task.done()
    with pytest.raises(RuntimeError, match="cancelled"):
        handle.result


@pytest.mark.asyncio
async def test_cancel_on_abandon_false_lets_pipeline_finish():
    pipeline = Pipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=3, chunk_delay=0.01))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}}, cancel_on_abandon=False)

    async def consumer() -> None:
        async for _ in handle:
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.015)  # consume ~1 chunk
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # the consumer was cancelled, but cancel_on_abandon=False leaves the pipeline running
    await handle._task
    assert not handle._task.cancelled()
    assert handle.result["streamer"] == {"reply": "s-final"}
