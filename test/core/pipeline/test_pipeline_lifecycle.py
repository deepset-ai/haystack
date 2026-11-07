# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

import pytest

from haystack import Pipeline, component


@component
class LifecycleRecorder:
    """Records every lifecycle method called, so tests can assert which ones the pipeline picks."""

    def __init__(self):
        self.events = []

    def warm_up(self):
        self.events.append("warm_up")

    async def warm_up_async(self):
        self.events.append("warm_up_async")

    @component.output_types(value=int)
    def run(self):
        self.events.append("run")
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        self.events.append("run_async")
        return {"value": 1}

    def close(self):
        self.events.append("close")

    async def close_async(self):
        self.events.append("close_async")


@component
class SyncOnlyRecorder:
    """Implements only the synchronous warm_up and close, to exercise the async fallbacks."""

    def __init__(self):
        self.events = []

    def warm_up(self):
        self.events.append("warm_up")

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}

    def close(self):
        self.events.append("close")


@component
class BareComponent:
    """Implements no lifecycle method at all."""

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}


class LoopBoundAsyncClient:
    """Mimics a real async client (aiohttp): binds to the loop it is created on and refuses any other."""

    def __init__(self):
        self._loop = asyncio.get_running_loop()

    async def use(self):
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("async client used on a different event loop than the one it was created on")


@component
class AsyncClientComponent:
    """Creates a loop-bound async client in warm_up_async and uses it in run_async."""

    def __init__(self):
        self.client: LoopBoundAsyncClient | None = None

    async def warm_up_async(self):
        if self.client is None:
            self.client = LoopBoundAsyncClient()

    @component.output_types(value=int)
    async def run_async(self):
        assert self.client is not None
        await self.client.use()
        return {"value": 1}

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}


async def test_run_async_uses_warm_up_async():
    """When a component implements warm_up_async, run_async uses it and does not also call its sync warm_up."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    await pipe.run_async({"rec": {}})
    assert "warm_up_async" in rec.events
    assert "warm_up" not in rec.events


async def test_warm_up_async_falls_back_to_sync_warm_up():
    """A component with only the sync warm_up is still warmed by run_async through that method."""
    rec = SyncOnlyRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    await pipe.run_async({"rec": {}})
    assert rec.events == ["warm_up"]


def test_sync_run_uses_sync_warm_up():
    """The sync run path warms components via the sync warm_up, never warm_up_async."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.run({"rec": {}})
    assert "warm_up" in rec.events
    assert "warm_up_async" not in rec.events


def test_pipeline_close_calls_sync_close_only():
    """Pipeline.close() calls each component's sync close, never close_async."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.close()
    assert "close" in rec.events
    assert "close_async" not in rec.events


async def test_pipeline_close_async_calls_async_close_only():
    """When a component implements close_async, Pipeline.close_async() uses it and does not also call its sync close."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    await pipe.close_async()
    assert "close_async" in rec.events
    assert "close" not in rec.events


async def test_close_async_falls_back_to_sync_close():
    """A component with only the sync close is still released by close_async through that method."""
    rec = SyncOnlyRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    await pipe.close_async()
    assert rec.events == ["close"]


async def test_run_does_not_auto_close():
    """Running a pipeline (sync or async) never closes components; closing is always explicit."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.run({"rec": {}})
    await pipe.run_async({"rec": {}})
    assert "close" not in rec.events
    assert "close_async" not in rec.events


async def test_lifecycle_methods_are_optional():
    """A component without lifecycle methods works: every call is hasattr-guarded and skipped."""
    pipe = Pipeline()
    pipe.add_component("bare", BareComponent())
    await pipe.warm_up_async()
    pipe.close()
    await pipe.close_async()
    await pipe.run_async({"bare": {}})


def test_loop_bound_client_rejects_other_loop():
    """The fake client raises when used from a different loop.

    This ensures the affinity test below enforces loop binding.
    """

    async def _make_loop_bound_client():
        return LoopBoundAsyncClient()

    client = asyncio.run(_make_loop_bound_client())
    with pytest.raises(RuntimeError):
        asyncio.run(client.use())


async def test_async_client_bound_to_run_loop():
    """warm_up_async creates the async client on the loop run_async uses, so it stays usable there."""
    pipe = Pipeline()
    pipe.add_component("client_component", AsyncClientComponent())
    await pipe.warm_up_async()
    # Would raise if warm_up_async had bound the client to a different loop than run_async
    await pipe.run_async({"client_component": {}})


@pytest.mark.parametrize("mode", ["sync", "async", "async_sync_fallback"])
@pytest.mark.parametrize("failure_indices", [(), (0,), (1,), (0, 1)])
async def test_close_releases_later_resources_after_component_error(
    mode: str, failure_indices: tuple[int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = Pipeline()
    errors = {index: OSError(f"cleanup failed for {index}") for index in failure_indices}
    events: list[int] = []

    def close_resource(index: int, resource: BinaryIO) -> None:
        events.append(index)
        resource.close()
        if index in errors:
            raise errors[index]

    async def close_resource_async(index: int, resource: BinaryIO) -> None:
        await asyncio.sleep(0)
        close_resource(index, resource)

    with ExitStack() as stack:
        resources = [stack.enter_context((tmp_path / str(index)).open("wb")) for index in range(3)]
        for index, resource in enumerate(resources):
            recorder = SyncOnlyRecorder() if mode == "async_sync_fallback" else LifecycleRecorder()
            if mode == "async":
                monkeypatch.setattr(recorder, "close_async", lambda i=index, r=resource: close_resource_async(i, r))
            else:
                monkeypatch.setattr(recorder, "close", lambda i=index, r=resource: close_resource(i, r))
            pipe.add_component(f"recorder_{index}", recorder)

        async def close_pipeline() -> None:
            if mode == "sync":
                pipe.close()
            else:
                await pipe.close_async()

        if errors:
            with pytest.raises(OSError, match=f"cleanup failed for {failure_indices[0]}") as caught:
                await close_pipeline()
            assert caught.value is errors[failure_indices[0]]
        else:
            await close_pipeline()

        assert events == [0, 1, 2]
        assert all(resource.closed for resource in resources)


@pytest.mark.parametrize("mode", ["sync", "async", "async_sync_fallback"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
async def test_close_does_not_suppress_base_exceptions(
    mode: str, error_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = Pipeline()
    first = SyncOnlyRecorder() if mode == "async_sync_fallback" else LifecycleRecorder()
    later = SyncOnlyRecorder() if mode == "async_sync_fallback" else LifecycleRecorder()
    error = error_type("stop cleanup")

    def interrupt_close() -> None:
        raise error

    async def interrupt_close_async() -> None:
        raise error

    if mode == "async":
        monkeypatch.setattr(first, "close_async", interrupt_close_async)
    else:
        monkeypatch.setattr(first, "close", interrupt_close)
    pipe.add_component("first", first)
    pipe.add_component("later", later)

    with pytest.raises(error_type, match="stop cleanup") as caught:
        if mode == "sync":
            pipe.close()
        else:
            await pipe.close_async()
    assert caught.value is error
    assert later.events == []
