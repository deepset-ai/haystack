# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from haystack import AsyncPipeline, component
from haystack.core.errors import PipelineRuntimeError


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


def test_non_async_component_error_handling():
    """Test that non-async components in AsyncPipeline properly wrap exceptions with PipelineRuntimeError"""
    
    @component
    class ErroringComponent:
        @component.output_types(output=str)
        def run(self, value: str):
            raise ValueError("Test error from non-async component")
    
    erroring_component = ErroringComponent()
    pp = AsyncPipeline()
    pp.add_component("erroring_component", erroring_component)
    
    inputs = {"value": "test_value"}
    
    async def run_pipeline():
        return await pp.run_async({"erroring_component": inputs})
    
    with pytest.raises(PipelineRuntimeError) as exc_info:
        asyncio.run(run_pipeline())
    
    # Verify that the error includes component information
    assert "Component name: 'erroring_component'" in str(exc_info.value)
    assert "Component type: 'ErroringComponent'" in str(exc_info.value)
    assert "Test error from non-async component" in str(exc_info.value)
