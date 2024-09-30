# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from math import log2
from typing import Any, Dict, List

from haystack import Document, component


@component
class DocumentNDCGEvaluator:
    """
    Evaluator that calculates the normalized discounted cumulative gain (NDCG) of retrieved documents.

    Each question can have multiple ground truth documents and multiple retrieved documents.
    If the ground truth documents have relevance scores, the NDCG calculation uses these scores.
    Otherwise, it uses the inverse of the document ranks as scores.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentNDCGEvaluator

    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France"), Document(content="Paris")]],
        retrieved_documents=[[Document(content="France"), Document(content="Germany"), Document(content="Paris")]],
    )
    print(result["individual_scores"])
    # [0.8869]
    print(result["score"])
    # 0.8869
    ```
    """

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Run the DocumentNDCGEvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question with relevance scores or sorted by relevance.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the NDCG for each question.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        individual_scores = []

        for gt_docs, ret_docs in zip(ground_truth_documents, retrieved_documents):
            dcg = self._calculate_dcg(gt_docs, ret_docs)
            idcg = self._calculate_idcg(gt_docs)
            ndcg = dcg / idcg if idcg > 0 else 0
            individual_scores.append(ndcg)

        score = sum(individual_scores) / len(ground_truth_documents)

        return {"score": score, "individual_scores": individual_scores}

    def _calculate_dcg(self, gt_docs: List[Document], ret_docs: List[Document]) -> float:
        dcg = 0
        id_to_score = {doc.id: doc.score for doc in gt_docs}
        for i, doc in enumerate(ret_docs):
            if doc.id in id_to_score:  # TODO Related to https://github.com/deepset-ai/haystack/issues/8412
                # If the gt document has a score, use it; otherwise, use the inverse of the rank
                relevance = id_to_score[doc.id] if id_to_score[doc.id] is not None else 1 / (i + 1)
                dcg += relevance / log2(i + 2)  # i + 2 because i is 0-indexed
        return dcg

    def _calculate_idcg(self, gt_docs: List[Document]) -> float:
        idcg = 0
        for i, doc in enumerate(sorted(gt_docs, key=lambda x: x.score if x.score is not None else 1, reverse=True)):
            # If the document has a score, use it; otherwise, use the inverse of the rank
            relevance = doc.score if doc.score is not None else 1 / (i + 1)
            idcg += relevance / log2(i + 2)
        return idcg
