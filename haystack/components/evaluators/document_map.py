# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

from haystack import Document, component


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

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
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
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high retrieved documents are ranked.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        individual_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            score = 0.0
            for ground_document in ground_truth:
                if ground_document.content is None:
                    continue

                average_precision = 0.0
                relevant_documents = 0

                for rank, retrieved_document in enumerate(retrieved):
                    if retrieved_document.content is None:
                        continue

                    if ground_document.content in retrieved_document.content:
                        relevant_documents += 1
                        average_precision += relevant_documents / (rank + 1)
                if relevant_documents > 0:
                    score = average_precision / relevant_documents
            individual_scores.append(score)

        score = sum(individual_scores) / len(retrieved_documents)

        return {"score": score, "individual_scores": individual_scores}
