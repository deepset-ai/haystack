# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import AsyncPipeline, component
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
        streaming_callback: AsyncStreamingCallbackT | None = None,
    ) -> None:
        self.prefix = prefix
        self.n_chunks = n_chunks
        self.fail = fail
        self.streaming_callback = streaming_callback

    @component.output_types(reply=str)
    def run(self, prompt: str, streaming_callback: AsyncStreamingCallbackT | None = None) -> dict:
        return {"reply": f"{self.prefix}-final"}

    @component.output_types(reply=str)
    async def run_async(self, prompt: str, streaming_callback: AsyncStreamingCallbackT | None = None) -> dict:
        for i in range(self.n_chunks):
            chunk = StreamingChunk(content=f"{self.prefix}{i}")
            if streaming_callback is not None:
                await streaming_callback(chunk)
        if self.fail:
            raise RuntimeError("boom")
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
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=3))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi"}})
    chunks = [c async for c in handle]

    assert [c.content for c in chunks] == ["s0", "s1", "s2"]
    assert handle.result["streamer"] == {"reply": "s-final"}


@pytest.mark.asyncio
async def test_stream_composes_with_init_streaming_callback():
    seen = []

    async def init_callback(chunk: StreamingChunk) -> None:
        seen.append(chunk.content)

    pipeline = AsyncPipeline()
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

    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=2, streaming_callback=init_callback))

    handle = pipeline.stream(data={"streamer": {"prompt": "hi", "streaming_callback": runtime_callback}})
    _ = [c async for c in handle]

    assert init_seen == []
    assert runtime_seen == ["s0", "s1"]


def test_stream_rejects_sync_runtime_callback():
    def sync_callback(chunk: StreamingChunk) -> None:
        pass

    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=2))

    with pytest.raises(ValueError, match="async"):
        pipeline.stream(data={"streamer": {"prompt": "hi", "streaming_callback": sync_callback}})


@pytest.mark.asyncio
async def test_stream_streams_all_components_by_default():
    pipeline = AsyncPipeline()
    pipeline.add_component("a", StreamingEcho(prefix="a", n_chunks=2))
    pipeline.add_component("b", StreamingEcho(prefix="b", n_chunks=2))

    handle = pipeline.stream(data={"a": {"prompt": "x"}, "b": {"prompt": "y"}})
    chunks = [c async for c in handle]

    assert {c.content for c in chunks} == {"a0", "a1", "b0", "b1"}


@pytest.mark.asyncio
async def test_stream_filters_to_selected_components():
    pipeline = AsyncPipeline()
    pipeline.add_component("a", StreamingEcho(prefix="a", n_chunks=2))
    pipeline.add_component("b", StreamingEcho(prefix="b", n_chunks=2))

    handle = pipeline.stream(data={"a": {"prompt": "x"}, "b": {"prompt": "y"}}, streaming_components=["a"])
    chunks = [c async for c in handle]

    assert {c.content for c in chunks} == {"a0", "a1"}
    assert handle.result["a"] == {"reply": "a-final"}
    assert handle.result["b"] == {"reply": "b-final"}


def test_stream_raises_for_unknown_component_in_filter():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho())

    with pytest.raises(ValueError, match="Unknown components"):
        pipeline.stream(data={"streamer": {"prompt": "x"}}, streaming_components=["typo"])


def test_stream_raises_for_non_streaming_component_in_filter():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho())
    pipeline.add_component("passthrough", Passthrough())

    with pytest.raises(ValueError, match="do not support streaming"):
        pipeline.stream(
            data={"streamer": {"prompt": "x"}, "passthrough": {"prompt": "x"}}, streaming_components=["passthrough"]
        )


@pytest.mark.asyncio
async def test_stream_propagates_pipeline_exception_during_iteration():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=1, fail=True))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    with pytest.raises(PipelineRuntimeError) as excinfo:
        async for _ in handle:
            pass

    # the pipeline wraps the original RuntimeError; the "boom" message is preserved in the chain
    chain = [excinfo.value, excinfo.value.__cause__, getattr(excinfo.value.__cause__, "__cause__", None)]
    assert any("boom" in str(e) for e in chain if e is not None)


@pytest.mark.asyncio
async def test_result_raises_after_failure_with_chained_cause():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=1, fail=True))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    with pytest.raises(PipelineRuntimeError):
        async for _ in handle:
            pass

    with pytest.raises(RuntimeError, match="Pipeline failed") as excinfo:
        handle.result
    assert excinfo.value.__cause__ is not None


@pytest.mark.asyncio
async def test_result_raises_when_pipeline_not_finished():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=5))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    # no await between `stream()` and reading `result`, so the task has not started yet
    with pytest.raises(RuntimeError, match="Pipeline has not finished"):
        handle.result

    await handle.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_pipeline_and_result_reports_cancelled():
    pipeline = AsyncPipeline()
    pipeline.add_component("streamer", StreamingEcho(prefix="s", n_chunks=100))

    handle = pipeline.stream(data={"streamer": {"prompt": "x"}})

    await handle.aclose()

    with pytest.raises(RuntimeError, match="Pipeline was cancelled"):
        handle.result
