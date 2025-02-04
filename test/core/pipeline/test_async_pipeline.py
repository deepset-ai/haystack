import asyncio

from haystack import AsyncPipeline


def test_async_pipeline_reentrance(waiting_component, spying_tracer):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    run_data = [{"wait_for": 1}, {"wait_for": 2}]

    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    try:
        async_loop.run_until_complete(run_all())
        component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run_async"]
        for span in component_spans:
            assert span.tags["haystack.component.visits"] == 1
    finally:
        async_loop.close()
