# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Optional

import pytest

from haystack import AsyncPipeline, component


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


def test_component_with_empty_dict_as_output_appears_in_results():
    """Test that components that return an empty dict as output appear in results as an empty dict"""

    @component
    class Producer:
        def __init__(self, prefix: str):
            self.prefix = prefix

        @component.output_types(value=Optional[str])
        def run(self, text: Optional[str]):
            return {"value": f"{self.prefix}: {text}"}

        @component.output_types(value=Optional[str])
        async def run_async(self, text: Optional[str]):
            return {"value": f"{self.prefix}: {text}"}

    @component
    class EmptyProcessor:
        @component.output_types()
        def run(self, sources: list[str]):
            # Returns empty dict when sources is empty
            return {}

        @component.output_types()
        async def run_async(self, sources: list[str]):
            # Returns empty dict when sources is empty
            return {}

    @component
    class Combiner:
        @component.output_types(combined=str)
        def run(self, input_a: Optional[str], input_b: Optional[str]):
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

        @component.output_types(combined=str)
        async def run_async(self, input_a: Optional[str], input_b: Optional[str]):
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

    pp = AsyncPipeline()
    pp.add_component("producer_a", Producer("A"))
    pp.add_component("producer_b", Producer("B"))
    pp.add_component("empty_processor", EmptyProcessor())
    pp.add_component("combiner", Combiner())

    pp.connect("producer_a.value", "combiner.input_a")
    pp.connect("producer_b.value", "combiner.input_b")

    result = pp.run(
        {"producer_a": {"text": "hello"}, "producer_b": {"text": "world"}, "empty_processor": {"sources": []}},
        include_outputs_from={"producer_a", "empty_processor", "combiner"},
    )

    # Producer A should appear in results because it's in include_outputs_from
    assert "producer_a" in result
    assert result["producer_a"] == {"value": "A: hello"}
    # Producer B should NOT appear since it's not in include_outputs_from
    assert "producer_b" not in result
    # Combiner should appear in results
    assert "combiner" in result
    assert result["combiner"] == {"combined": "A: hello | B: world"}
    # Empty processor should appear in results even though it returns an empty dict
    # because it's in include_outputs_from
    assert "empty_processor" in result
    assert result["empty_processor"] == {}
