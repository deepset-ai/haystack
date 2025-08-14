# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from haystack import AsyncPipeline


def test_async_pipeline_reentrance(waiting_component, spying_tracer):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    run_data = [{"wait_for": 0.001}, {"wait_for": 0.002}]

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run_async"]
    for span in component_spans:
        assert span.tags["haystack.component.visits"] == 1


def test_run_in_sync_context(waiting_component):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    result = pp.run({"wait_for": 0.001})

    assert result == {"wait": {"waited_for": 0.001}}


def test_run_in_async_context_raises_runtime_error():
    pp = AsyncPipeline()

    async def call_run():
        pp.run({})

    with pytest.raises(RuntimeError, match="Cannot call run\\(\\) from within an async context"):
        asyncio.run(call_run())
