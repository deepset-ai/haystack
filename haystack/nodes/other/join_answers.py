from typing import Optional, List, Dict, Tuple

from haystack.schema import Answer
from haystack.nodes import BaseComponent


class JoinAnswers(BaseComponent):
    """
    A node to join `Answer`s produced by multiple `Reader` nodes.
    """

    def __init__(
        self, join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None
    ):
        """
        :param join_mode: `"concatenate"` to combine documents from multiple `Reader`s. `"merge"` to aggregate scores
        of individual `Answer`s.
        :param weights: A node-wise list (length of list must be equal to the number of input nodes) of weights for
            adjusting `Answer` scores when using the `"merge"` join_mode. By default, equal weight is assigned to each
            `Reader` score. This parameter is not compatible with the `"concatenate"` join_mode.
        :param top_k_join: Limit `Answer`s to top_k based on the resulting scored of the join.
        """

        assert join_mode in ["concatenate", "merge"], f"JoinAnswers node does not support '{join_mode}' join_mode."
        assert not (
            weights is not None and join_mode == "concatenate"
        ), "Weights are not compatible with 'concatenate' join_mode"

        super().__init__()

        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join

    def run(self, inputs: List[Dict], top_k_join: Optional[int] = None) -> Tuple[Dict, str]:  # type: ignore
        reader_results = [inp["answers"] for inp in inputs]

        if not top_k_join:
            top_k_join = self.top_k_join

        if self.join_mode == "concatenate":
            concatenated_answers = [answer for cur_reader_result in reader_results for answer in cur_reader_result]
            concatenated_answers = sorted(concatenated_answers, reverse=True)[:top_k_join]
            return {"answers": concatenated_answers, "labels": inputs[0].get("labels", None)}, "output_1"

        elif self.join_mode == "merge":
            merged_answers = self._merge_answers(reader_results)

            merged_answers = merged_answers[:top_k_join]
            return {"answers": merged_answers, "labels": inputs[0].get("labels", None)}, "output_1"

        else:
            raise ValueError(f"Invalid join_mode: {self.join_mode}")

    def _merge_answers(self, reader_results: List[List[Answer]]) -> List[Answer]:
        weights = self.weights if self.weights else [1 / len(reader_results)] * len(reader_results)

        for result, weight in zip(reader_results, weights):
            for answer in result:
                if isinstance(answer.score, float):
                    answer.score *= weight

        return sorted([answer for cur_reader_result in reader_results for answer in cur_reader_result], reverse=True)
