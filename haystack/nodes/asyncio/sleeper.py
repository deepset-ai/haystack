from typing import Optional, List, Dict, Union, Any, Literal
import asyncio

import numpy as np

from haystack.schema import Document, MultiLabel, Answer
from haystack.nodes.base import BaseComponent


class Sleeper(BaseComponent):
    """
    Simple component that sleeps for a random amount of time and then returns a dummy answer.
    """

    outgoing_edges: int = 1

    def __init__(
        self,
        mean_sleep_in_seconds: float = 10,
        sleep_scale: float = 1.0,
        answer_type: Literal["generative", "extractive", "other"] = "generative",
        answer_score: Optional[float] = None,
        answer: str = "Placeholder",
    ) -> None:
        super().__init__()
        self._mean_sleep_in_seconds = mean_sleep_in_seconds
        self._sleep_scale = sleep_scale
        self._answer_type = answer_type
        self._answer = answer
        self._answer_score = answer_score

    # pylint: disable=invalid-overridden-method
    async def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ):
        if query is None:
            return {"answers": []}, "output_1"

        meta_data = meta if meta is not None else {}

        # Sleep to mock processing time
        sleep_time_seconds = max(0.0, np.random.normal(self._mean_sleep_in_seconds, self._sleep_scale))
        await asyncio.sleep(sleep_time_seconds)

        return {
            "answers": [Answer(answer=self._answer, type=self._answer_type, meta=meta_data, score=self._answer_score)]
        }, "output_1"

    # pylint: disable=too-many-arguments
    def run_batch(  # type: ignore
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        queries = queries or []
        query_list: List[str] = [queries] if isinstance(queries, str) else queries
        result: Dict[Any, Any] = {"answers": [], "queries": []}

        for query in query_list:
            iteration_result, _ = self.run(query=query)  # type: ignore
            result["answers"].append(iteration_result["answers"])  # type: ignore
            result["queries"].append(query)

        return result, "output_1"
