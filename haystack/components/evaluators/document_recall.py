from enum import Enum
from typing import Any, Dict, List, Union

from haystack.core.component import component
from haystack.dataclasses import Document


class RecallMode(Enum):
    """
    Enum for the mode to use for calculating the recall score.
    """

    # Score is based on whether any document is retrieved.
    SINGLE_HIT = "single_hit"
    # Score is based on how many documents were retrieved.
    MULTI_HIT = "multi_hit"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "RecallMode":
        enum_map = {e.value: e for e in RecallMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown recall mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class DocumentRecallEvaluator:
    """
    Evaluator that calculates the Recall score for a list of questions.
    Returns both a list of scores for each question and the average.
    Each question can have multiple ground truth documents and multiple predicted documents.

    Usage example:
    ```python
    from haystack.components.evaluators import DocumentRecallEvaluator
    evaluator = DocumentRecallEvaluator()
    result = evaluator.run(
        questions=["What is the capital of Germany?", "What is the capital of France?"],
        ground_truth_answers=[["Berlin"], ["Paris"]],
        predicted_answers=[["Paris"], ["London"]],
    )
    print(result["individual_scores"])
    # [0.0, 0.0]
    print(result["score"])
    # 0.0
    ```
    """

    def __init__(self, mode: Union[str, RecallMode] = RecallMode.SINGLE_HIT):
        """
        Create a DocumentRecallEvaluator component.

        :param mode:
            Mode to use for calculating the recall score.
        """
        if isinstance(mode, str):
            mode = RecallMode.from_str(mode)

        mode_functions = {RecallMode.SINGLE_HIT: self._recall_single_hit, RecallMode.MULTI_HIT: self._recall_multi_hit}
        self.mode_function = mode_functions[mode]

    def _recall_single_hit(self, ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> bool:
        unique_truths = {g.content for g in ground_truth_documents}
        unique_retrievals = {p.content for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        return len(retrieved_ground_truths) > 0

    def _recall_multi_hit(self, ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> float:
        unique_truths = {g.content for g in ground_truth_documents}
        unique_retrievals = {p.content for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        return len(retrieved_ground_truths) / len(ground_truth_documents)

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self,
        questions: List[str],
        ground_truth_documents: List[List[Document]],
        retrieved_documents: List[List[Document]],
    ) -> Dict[str, Any]:
        """
        Run the DocumentRecallEvaluator on the given inputs.
        All lists must have the same length.

        :param questions:
            A list of questions.
        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `invididual_scores` - A list of numbers from 0.0 to 1.0 that represents the proportion of matching documents retrieved.
                                    If the mode is `single_hit`, the individual scores are True or False.
        """
        if not len(questions) == len(ground_truth_documents) == len(retrieved_documents):
            msg = "The length of questions, ground_truth_documents, and predicted_documents must be the same."
            raise ValueError(msg)

        scores = []
        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            score = self.mode_function(ground_truth, retrieved)
            scores.append(score)

        return {"score": sum(scores) / len(questions), "individual_scores": scores}
