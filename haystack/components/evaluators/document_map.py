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

    # Refer to https://www.pinecone.io/learn/offline-evaluation/ for the algorithm.
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

            ground_truth_contents = [doc.content for doc in ground_truth if doc.content is not None]
            for rank, retrieved_document in enumerate(retrieved):
                if retrieved_document.content is None:
                    continue

                if retrieved_document.content in ground_truth_contents:
                    relevant_documents += 1
                    average_precision_numerator += relevant_documents / (rank + 1)
            if relevant_documents > 0:
                average_precision = average_precision_numerator / relevant_documents
            individual_scores.append(average_precision)

        score = sum(individual_scores) / len(ground_truth_documents)
        return {"score": score, "individual_scores": individual_scores}
