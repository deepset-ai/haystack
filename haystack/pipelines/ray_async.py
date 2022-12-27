from __future__ import annotations
from typing import Any, Dict, Tuple

from haystack.pipelines.ray import RayPipeline


class AsyncRayPipeline(RayPipeline):
    async def _run_node(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:  # type: ignore
        # Async calling of Ray Deployments instead of using `ray.get()` as in thr sync version of `_run_node()`
        # in the `RayPipeline` class in `ray.py`.
        # See https://docs.ray.io/en/latest/ray-core/actors/async_api.html#objectrefs-as-asyncio-futures

        # return ray.get(self.graph.nodes[node_id]["component"].remote(**node_input))
        return await self.graph.nodes[node_id]["component"].remote(**node_input)
