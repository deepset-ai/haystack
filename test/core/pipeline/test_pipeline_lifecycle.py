# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

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


@component
class FailingCloseComponent:
    """Raises an error during close and close_async while recording that close was attempted."""

    def __init__(self, error: Exception):
        self.error = error
        self.events: list[str] = []

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}

    def close(self):
        self.events.append("close")
        raise self.error

    async def close_async(self):
        self.events.append("close_async")
        raise self.error


@component
class FailingSyncOnlyCloseComponent:
    """Implements only a failing sync close, to exercise sync-fallback error handling in close_async."""

    def __init__(self, error: Exception):
        self.error = error
        self.events: list[str] = []

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}

    def close(self):
        self.events.append("close")
        raise self.error


async def test_async_client_bound_to_run_loop():
    """warm_up_async creates the async client on the loop run_async uses, so it stays usable there."""
    pipe = Pipeline()
    pipe.add_component("client_component", AsyncClientComponent())
    await pipe.warm_up_async()
    # Would raise if warm_up_async had bound the client to a different loop than run_async
    await pipe.run_async({"client_component": {}})


def test_pipeline_close_continues_after_error_and_raises_first():
    """Pipeline.close() attempts every component even when earlier ones raise, then re-raises the first error."""
    first_failing = FailingCloseComponent(RuntimeError("first close error"))
    healthy = LifecycleRecorder()
    second_failing = FailingCloseComponent(ValueError("second close error"))
    last_healthy = SyncOnlyRecorder()

    pipe = Pipeline()
    pipe.add_component("first_failing", first_failing)
    pipe.add_component("healthy", healthy)
    pipe.add_component("second_failing", second_failing)
    pipe.add_component("last_healthy", last_healthy)

    with pytest.raises(RuntimeError, match="first close error"):
        pipe.close()

    assert first_failing.events == ["close"]
    assert healthy.events == ["close"]
    assert second_failing.events == ["close"]
    assert last_healthy.events == ["close"]


async def test_pipeline_close_async_continues_after_error_and_raises_first():
    """Pipeline.close_async() attempts all async and sync-fallback components, then re-raises the first error."""
    first_failing = FailingCloseComponent(RuntimeError("first async close error"))
    healthy_async = LifecycleRecorder()
    failing_sync = FailingSyncOnlyCloseComponent(ValueError("sync fallback close error"))
    healthy_sync = SyncOnlyRecorder()

    pipe = Pipeline()
    pipe.add_component("first_failing", first_failing)
    pipe.add_component("healthy_async", healthy_async)
    pipe.add_component("failing_sync", failing_sync)
    pipe.add_component("healthy_sync", healthy_sync)

    with pytest.raises(RuntimeError, match="first async close error"):
        await pipe.close_async()

    assert first_failing.events == ["close_async"]
    assert healthy_async.events == ["close_async"]
    assert failing_sync.events == ["close"]
    assert healthy_sync.events == ["close"]
