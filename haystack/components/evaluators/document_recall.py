# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any

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

    def __init__(self, mode: str | RecallMode = RecallMode.SINGLE_HIT, document_comparison_field: str = "content"):
        """
        Create a DocumentRecallEvaluator component.

        :param mode:
            Mode to use for calculating the recall score.
        :param document_comparison_field:
            The Document field to use for comparison. Possible options:
            - `"content"`: uses `doc.content`
            - `"id"`: uses `doc.id`
            - A `meta.` prefix followed by a key name: uses `doc.meta["<key>"]`
              (e.g. `"meta.file_id"`, `"meta.page_number"`)
              Nested keys are supported (e.g. `"meta.source.url"`).
        """
        if isinstance(mode, str):
            mode = RecallMode.from_str(mode)

        self.mode = mode
        self.document_comparison_field = document_comparison_field

    def _get_comparison_value(self, doc: Document) -> Any:
        """
        Extract the comparison value from a document based on the configured field.
        """
        if self.document_comparison_field == "content":
            return doc.content
        if self.document_comparison_field == "id":
            return doc.id
        if self.document_comparison_field.startswith("meta."):
            parts = self.document_comparison_field[5:].split(".")
            value = doc.meta
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return None
                value = value[part]
            return value
        msg = (
            f"Unsupported document_comparison_field: '{self.document_comparison_field}'. "
            "Use 'content', 'id', or 'meta.<key>'."
        )
        raise ValueError(msg)

    def _recall_single_hit(self, ground_truth_documents: list[Document], retrieved_documents: list[Document]) -> float:
        unique_truths = {self._get_comparison_value(g) for g in ground_truth_documents}
        unique_retrievals = {self._get_comparison_value(p) for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        return float(len(retrieved_ground_truths) > 0)

    def _recall_multi_hit(self, ground_truth_documents: list[Document], retrieved_documents: list[Document]) -> float:
        unique_truths = {self._get_comparison_value(g) for g in ground_truth_documents}
        unique_retrievals = {self._get_comparison_value(p) for p in retrieved_documents}
        retrieved_ground_truths = unique_truths.intersection(unique_retrievals)

        if not unique_truths or unique_truths <= {"", None}:
            logger.warning(
                "There are no ground truth documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )
            return 0.0

        if not unique_retrievals or unique_retrievals <= {"", None}:
            logger.warning(
                "There are no retrieved documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )
            return 0.0

        return len(retrieved_ground_truths) / len(unique_truths)

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self, ground_truth_documents: list[list[Document]], retrieved_documents: list[list[Document]]
    ) -> dict[str, Any]:
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

        if self.mode == RecallMode.SINGLE_HIT:
            mode_function = self._recall_single_hit
        elif self.mode == RecallMode.MULTI_HIT:
            mode_function = self._recall_multi_hit

        scores = [mode_function(gt, ret) for gt, ret in zip(ground_truth_documents, retrieved_documents)]

        return {"score": sum(scores) / len(retrieved_documents), "individual_scores": scores}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, mode=str(self.mode), document_comparison_field=self.document_comparison_field)
