# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_to_dict


@component
class DocumentMAPEvaluator:
    """
    A Mean Average Precision (MAP) evaluator for documents.

    Evaluator that calculates the mean average precision of the retrieved documents, a metric
    that measures how high retrieved documents are ranked.
    Each question can have multiple ground truth documents and multiple retrieved documents.

    `DocumentMAPEvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
    should be used to clean and normalize the documents before passing them to this evaluator.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentMAPEvaluator

    evaluator = DocumentMAPEvaluator()
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
    # [1.0, 0.8333333333333333]
    print(result["score"])
    # 0.9166666666666666
    ```
    """

    def __init__(self, document_comparison_field: str = "content"):
        """
        Create a DocumentMAPEvaluator component.

        :param document_comparison_field:
            The Document field to use for comparison. Possible options:
            - `"content"`: uses `doc.content`
            - `"id"`: uses `doc.id`
            - A `meta.` prefix followed by a key name: uses `doc.meta["<key>"]`
              (e.g. `"meta.file_id"`, `"meta.page_number"`)
              Nested keys are supported (e.g. `"meta.source.url"`).
        """
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, document_comparison_field=self.document_comparison_field)

    # Refer to https://www.pinecone.io/learn/offline-evaluation/ for the algorithm.
    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self, ground_truth_documents: list[list[Document]], retrieved_documents: list[list[Document]]
    ) -> dict[str, Any]:
        """
        Run the DocumentMAPEvaluator on the given inputs.

        All lists must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high retrieved documents
                are ranked.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        individual_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            average_precision = 0.0
            average_precision_numerator = 0.0
            relevant_documents = 0

            ground_truth_values = [val for doc in ground_truth if (val := self._get_comparison_value(doc)) is not None]
            for rank, retrieved_document in enumerate(retrieved):
                retrieved_value = self._get_comparison_value(retrieved_document)
                if retrieved_value is None:
                    continue

                if retrieved_value in ground_truth_values:
                    relevant_documents += 1
                    average_precision_numerator += relevant_documents / (rank + 1)
            if relevant_documents > 0:
                average_precision = average_precision_numerator / relevant_documents
            individual_scores.append(average_precision)

        score = sum(individual_scores) / len(ground_truth_documents)
        return {"score": score, "individual_scores": individual_scores}
