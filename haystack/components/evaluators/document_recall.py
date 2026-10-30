# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any

from haystack import component, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.misc import _make_metadata_value_hashable

logger = logging.getLogger(__name__)


class RecallMode(Enum):
    """
    Enum for the mode to use for calculating the recall score.
    """

    # Score is based on whether any document is retrieved.
    SINGLE_HIT = "single_hit"
    # Score is based on how many documents were retrieved.
    MULTI_HIT = "multi_hit"

    def __str__(self) -> str:
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

    Set `include_diagnostics=True` to also return, for each question, which ground truth
    documents were missed (`missing_documents`), the fraction of ground truth documents
    retrieved (`support_coverage`), and the fraction of questions for which every ground
    truth document was retrieved (`all_support_rate`).
    """

    def __init__(
        self,
        mode: str | RecallMode = RecallMode.SINGLE_HIT,
        document_comparison_field: str = "content",
        include_diagnostics: bool = False,
    ) -> None:
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
              Values may also be lists or dictionaries.
        :param include_diagnostics:
            If `True`, `run` additionally returns per-query diagnostics: the comparison values of the
            ground truth documents that were not retrieved (`missing_documents`), the fraction of ground
            truth documents retrieved for each query (`support_coverage`), and the fraction of queries
            for which every ground truth document was retrieved (`all_support_rate`). The diagnostics
            use the same comparison values as the recall score, as configured by
            `document_comparison_field`. Defaults to `False`, which preserves the original outputs.
        """
        if isinstance(mode, str):
            mode = RecallMode.from_str(mode)

        self.mode = mode
        self.document_comparison_field = document_comparison_field
        self.include_diagnostics = include_diagnostics

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
            return _make_metadata_value_hashable(value)
        msg = (
            f"Unsupported document_comparison_field: '{self.document_comparison_field}'. "
            "Use 'content', 'id', or 'meta.<key>'."
        )
        raise ValueError(msg)

    def _unique_comparison_values(self, documents: list[Document]) -> set:
        """
        Collect the unique comparison values of the documents, ignoring missing or empty ones.
        """
        return {value for doc in documents if (value := self._get_comparison_value(doc)) not in ("", None)}

    def _comparison_sets(
        self, ground_truth_documents: list[Document], retrieved_documents: list[Document]
    ) -> tuple[set, set]:
        """
        Collect the unique comparison values of both sides, warning if either has no value to match on.
        """
        unique_truths = self._unique_comparison_values(ground_truth_documents)
        if not unique_truths:
            logger.warning(
                "There are no ground truth documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )

        unique_retrievals = self._unique_comparison_values(retrieved_documents)
        if not unique_retrievals:
            logger.warning(
                "There are no retrieved documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )

        return unique_truths, unique_retrievals

    def _recall_single_hit(self, ground_truth_documents: list[Document], retrieved_documents: list[Document]) -> float:
        unique_truths, unique_retrievals = self._comparison_sets(ground_truth_documents, retrieved_documents)
        return self._recall_single_hit_from_sets(unique_truths, unique_retrievals)

    @staticmethod
    def _recall_single_hit_from_sets(unique_truths: set, unique_retrievals: set) -> float:
        if not unique_truths or not unique_retrievals:
            return 0.0

        return float(len(unique_truths.intersection(unique_retrievals)) > 0)

    def _recall_multi_hit(self, ground_truth_documents: list[Document], retrieved_documents: list[Document]) -> float:
        unique_truths, unique_retrievals = self._comparison_sets(ground_truth_documents, retrieved_documents)
        return self._recall_multi_hit_from_sets(unique_truths, unique_retrievals)

    @staticmethod
    def _recall_multi_hit_from_sets(unique_truths: set, unique_retrievals: set) -> float:
        if not unique_truths or not unique_retrievals:
            return 0.0

        return len(unique_truths.intersection(unique_retrievals)) / len(unique_truths)

    def _missing_comparison_values(self, ground_truth_documents: list[Document], unique_retrievals: set) -> list[Any]:
        """
        Collect the comparison values of the ground truth documents absent from the retrieved documents.

        Values are deduplicated and returned in ground truth order so the output is deterministic.
        """
        missing = []
        seen = set()
        for doc in ground_truth_documents:
            value = self._get_comparison_value(doc)
            if value in ("", None) or value in seen:
                continue
            seen.add(value)
            if value not in unique_retrievals:
                missing.append(value)
        return missing

    @component.output_types(
        score=float,
        individual_scores=list[float],
        missing_documents=list[list[Any]],
        support_coverage=list[float],
        all_support_rate=float,
    )
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
            - `missing_documents` - Only returned if `include_diagnostics` is `True`. A list with one entry
                for each question, containing the comparison values of the ground truth documents that were
                not retrieved, in ground truth order.
            - `support_coverage` - Only returned if `include_diagnostics` is `True`. A list of numbers from
                0.0 to 1.0 that represents the fraction of ground truth documents retrieved for each question,
                regardless of the configured recall `mode`.
            - `all_support_rate` - Only returned if `include_diagnostics` is `True`. The fraction of
                questions for which every ground truth document was retrieved.
        """
        if len(ground_truth_documents) == 0 or len(retrieved_documents) == 0:
            msg = "ground_truth_documents and retrieved_documents must be provided."
            raise ValueError(msg)

        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        if self.mode == RecallMode.SINGLE_HIT:
            mode_function_from_sets = self._recall_single_hit_from_sets
        elif self.mode == RecallMode.MULTI_HIT:
            mode_function_from_sets = self._recall_multi_hit_from_sets

        scores = []
        missing_documents: list[list[Any]] = []
        support_coverage: list[float] = []
        all_support_count = 0
        for gt, ret in zip(ground_truth_documents, retrieved_documents, strict=True):
            unique_truths, unique_retrievals = self._comparison_sets(gt, ret)
            scores.append(mode_function_from_sets(unique_truths, unique_retrievals))
            if self.include_diagnostics:
                missing_documents.append(self._missing_comparison_values(gt, unique_retrievals))
                support_coverage.append(
                    len(unique_truths.intersection(unique_retrievals)) / len(unique_truths) if unique_truths else 0.0
                )
                if unique_truths and unique_truths.issubset(unique_retrievals):
                    all_support_count += 1

        result: dict[str, Any] = {"score": sum(scores) / len(retrieved_documents), "individual_scores": scores}
        if self.include_diagnostics:
            result["missing_documents"] = missing_documents
            result["support_coverage"] = support_coverage
            result["all_support_rate"] = all_support_count / len(retrieved_documents)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            mode=str(self.mode),
            document_comparison_field=self.document_comparison_field,
            include_diagnostics=self.include_diagnostics,
        )
