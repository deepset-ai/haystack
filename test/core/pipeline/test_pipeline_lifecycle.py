# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

from haystack import Pipeline, component


def _running_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


@component
class LifecycleRecorder:
    def __init__(self):
        self.events = []

    def warm_up(self):
        self.events.append(("warm_up", threading.get_ident(), _running_loop()))

    async def warm_up_async(self):
        self.events.append(("warm_up_async", threading.get_ident(), _running_loop()))

    @component.output_types(value=int)
    def run(self):
        self.events.append(("run", threading.get_ident(), _running_loop()))
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        self.events.append(("run_async", threading.get_ident(), _running_loop()))
        return {"value": 1}

    def close(self):
        self.events.append(("close", threading.get_ident(), _running_loop()))

    async def close_async(self):
        self.events.append(("close_async", threading.get_ident(), _running_loop()))


@component
class SyncWarmUpRecorder:
    def __init__(self):
        self.events = []

    def warm_up(self):
        self.events.append(("warm_up", threading.get_ident(), _running_loop()))

    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}

    def close(self):
        self.events.append(("close", threading.get_ident(), _running_loop()))


@component
class BareComponent:
    @component.output_types(value=int)
    def run(self):
        return {"value": 1}

    @component.output_types(value=int)
    async def run_async(self):
        return {"value": 1}


def test_run_async_uses_warm_up_async():
    """run_async warms components via warm_up_async (not the sync warm_up), on a running loop."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    asyncio.run(pipe.run_async({"rec": {}}))
    kinds = [event[0] for event in rec.events]
    assert "warm_up_async" in kinds
    assert "warm_up" not in kinds
    warm_up_event = next(event for event in rec.events if event[0] == "warm_up_async")
    assert warm_up_event[2] is not None


def test_sync_warm_up_fallback_blocks_on_loop_thread():
    """
    If a component implements only the sync warm_up, run_async still warms it by calling that sync
    warm_up directly on the event-loop thread, instead of offloading it to a worker thread.
    """
    rec = SyncWarmUpRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    asyncio.run(pipe.run_async({"rec": {}}))
    assert len(rec.events) == 1
    name, thread_id, loop = rec.events[0]
    assert name == "warm_up"
    assert loop is not None
    assert thread_id == threading.get_ident()


def test_sync_run_uses_sync_warm_up():
    """The sync run path warms components via sync warm_up, never warm_up_async."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.run({"rec": {}})
    kinds = [event[0] for event in rec.events]
    assert "warm_up" in kinds
    assert "warm_up_async" not in kinds


def test_async_lifecycle_shares_one_loop():
    """warm_up_async, run_async and close_async all run on the same event loop (correct async-client affinity)."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)

    async def drive():
        await pipe.warm_up_async()
        await pipe.run_async({"rec": {}})
        await pipe.close_async()

    asyncio.run(drive())
    loops = {event[0]: event[2] for event in rec.events}
    assert loops["warm_up_async"] is not None
    assert loops["warm_up_async"] is loops["run_async"]
    assert loops["run_async"] is loops["close_async"]


def test_pipeline_close_calls_sync_close_only():
    """Pipeline.close() calls each component's sync close, never close_async."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.close()
    kinds = [event[0] for event in rec.events]
    assert "close" in kinds
    assert "close_async" not in kinds


def test_pipeline_close_async_calls_async_close_only():
    """Pipeline.close_async() calls each component's close_async only, never the sync close."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    asyncio.run(pipe.close_async())
    kinds = [event[0] for event in rec.events]
    assert "close_async" in kinds
    assert "close" not in kinds


def test_close_async_falls_back_to_sync_close():
    """close_async closes a component that has only sync close, on the event-loop thread."""
    rec = SyncWarmUpRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    asyncio.run(pipe.close_async())
    assert len(rec.events) == 1
    name, thread_id, loop = rec.events[0]
    assert name == "close"
    assert loop is not None
    assert thread_id == threading.get_ident()


def test_run_does_not_auto_close():
    """Running a pipeline (sync or async) never auto-closes components; close is always explicit."""
    rec = LifecycleRecorder()
    pipe = Pipeline()
    pipe.add_component("rec", rec)
    pipe.run({"rec": {}})
    asyncio.run(pipe.run_async({"rec": {}}))
    kinds = [event[0] for event in rec.events]
    assert "close" not in kinds
    assert "close_async" not in kinds


def test_lifecycle_methods_are_optional():
    """
    A component that implements none of the lifecycle methods works fine: the pipeline guards every
    warm_up_async / close / close_async call with hasattr, so they are skipped instead of raising.
    """
    pipe = Pipeline()
    pipe.add_component("bare", BareComponent())
    asyncio.run(pipe.warm_up_async())
    pipe.close()
    asyncio.run(pipe.close_async())
    asyncio.run(pipe.run_async({"bare": {}}))
