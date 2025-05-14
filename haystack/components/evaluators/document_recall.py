# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Union

from haystack import component, default_to_dict, logging
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


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
        """
        Convert a string to a RecallMode enum.
        """
        enum_map = {e.value: e for e in RecallMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown recall mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class DocumentRecallEvaluator:
    """
    Evaluator that calculates the Recall score for a list of documents.

    Returns both a list of scores for each question and the average.
    There can be multiple ground truth documents and multiple predicted documents as input.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentRecallEvaluator

    evaluator = DocumentRecallEvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 1.0]
    print(result["score"])
    # 1.0
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

        mode_functions = {
            RecallMode.SINGLE_HIT: DocumentRecallEvaluator._recall_single_hit,
            RecallMode.MULTI_HIT: DocumentRecallEvaluator._recall_multi_hit,
        }
        self.mode_function = mode_functions[mode]
        self.mode = mode

    @staticmethod
    def _recall_single_hit(ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> float:
        unique_truths = {g.content for g in ground_truth_documents}
        unique_retrievals = {p.content for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        return float(len(retrieved_ground_truths) > 0)

    @staticmethod
    def _recall_multi_hit(ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> float:
        unique_truths = {g.content for g in ground_truth_documents}
        unique_retrievals = {p.content for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        if not unique_truths or unique_truths == {""}:
            logger.warning(
                "There are no ground truth documents or all of them have an empty string as content. "
                "Score will be set to 0."
            )
            return 0.0

        if not unique_retrievals or unique_retrievals == {""}:
            logger.warning(
                "There are no retrieved documents or all of them have an empty string as content. "
                "Score will be set to 0."
            )
            return 0.0

        return len(retrieved_ground_truths) / len(unique_truths)

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Run the DocumentRecallEvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the proportion of matching
                documents retrieved. If the mode is `single_hit`, the individual scores are 0 or 1.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        scores = []
        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            score = self.mode_function(ground_truth, retrieved)
            scores.append(score)

        return {"score": sum(scores) / len(retrieved_documents), "individual_scores": scores}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, mode=str(self.mode))
