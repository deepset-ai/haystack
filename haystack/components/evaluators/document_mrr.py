# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_to_dict


@component
class DocumentMRREvaluator:
    """
    Evaluator that calculates the mean reciprocal rank of the retrieved documents.

    MRR measures how high the first retrieved document is ranked.
    Each question can have multiple ground truth documents and multiple retrieved documents.

    `DocumentMRREvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
    should be used to clean and normalize the documents before passing them to this evaluator.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentMRREvaluator

    evaluator = DocumentMRREvaluator()
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

    def __init__(self, document_comparison_field: str = "content"):
        """
        Create a DocumentMRREvaluator component.

        :param document_comparison_field:
            The Document field to use for comparison. Possible options:
            - ``"content"``: uses ``doc.content``
            - ``"id"``: uses ``doc.id``
            - A ``meta.`` prefix followed by a key name: uses ``doc.meta["<key>"]``
              (e.g. ``"meta.file_id"``, ``"meta.page_number"``)
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
            meta_key = self.document_comparison_field[5:]
            return doc.meta.get(meta_key)
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
        Run the DocumentMRREvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high the first retrieved
                document is ranked.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        individual_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            reciprocal_rank = 0.0

            ground_truth_values = [
                self._get_comparison_value(doc) for doc in ground_truth if self._get_comparison_value(doc) is not None
            ]
            for rank, retrieved_document in enumerate(retrieved):
                retrieved_value = self._get_comparison_value(retrieved_document)
                if retrieved_value is None:
                    continue
                if retrieved_value in ground_truth_values:
                    reciprocal_rank = 1 / (rank + 1)
                    break
            individual_scores.append(reciprocal_rank)

        score = sum(individual_scores) / len(ground_truth_documents)

        return {"score": score, "individual_scores": individual_scores}
