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
    Otherwise, it assumes binary relevance of all ground truth documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentNDCGEvaluator

    evaluator = DocumentNDCGEvaluator()
    result = evaluator.run(
        ground_truth_documents=[[Document(content="France", score=1.0), Document(content="Paris", score=0.5)]],
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
        The list items within `ground_truth_documents` and `retrieved_documents` can differ in length.

        :param ground_truth_documents:
            Lists of expected documents, one list per question. Binary relevance is used if documents have no scores.
        :param retrieved_documents:
            Lists of retrieved documents, one list per question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the NDCG for each question.
        """
        self.validate_inputs(ground_truth_documents, retrieved_documents)

        individual_scores = []

        for gt_docs, ret_docs in zip(ground_truth_documents, retrieved_documents):
            dcg = self.calculate_dcg(gt_docs, ret_docs)
            idcg = self.calculate_idcg(gt_docs)
            ndcg = dcg / idcg if idcg > 0 else 0
            individual_scores.append(ndcg)

        score = sum(individual_scores) / len(ground_truth_documents)

        return {"score": score, "individual_scores": individual_scores}

    @staticmethod
    def validate_inputs(gt_docs: List[List[Document]], ret_docs: List[List[Document]]):
        """
        Validate the input parameters.

        :param gt_docs:
            The ground_truth_documents to validate.
        :param ret_docs:
            The retrieved_documents to validate.

        :raises ValueError:
            If the ground_truth_documents or the retrieved_documents are an empty a list.
            If the length of ground_truth_documents and retrieved_documents differs.
            If any list of documents in ground_truth_documents contains a mix of documents with and without a score.
        """
        if len(gt_docs) == 0 or len(ret_docs) == 0:
            msg = "ground_truth_documents and retrieved_documents must be provided."
            raise ValueError(msg)

        if len(gt_docs) != len(ret_docs):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        for docs in gt_docs:
            if any(doc.score is not None for doc in docs) and any(doc.score is None for doc in docs):
                msg = "Either none or all documents in each list of ground_truth_documents must have a score."
                raise ValueError(msg)

    @staticmethod
    def calculate_dcg(gt_docs: List[Document], ret_docs: List[Document]) -> float:
        """
        Calculate the discounted cumulative gain (DCG) of the retrieved documents.

        :param gt_docs:
            The ground truth documents.
        :param ret_docs:
            The retrieved documents.
        :returns:
            The discounted cumulative gain (DCG) of the retrieved
            documents based on the ground truth documents.
        """
        dcg = 0.0
        relevant_id_to_score = {doc.id: doc.score if doc.score is not None else 1 for doc in gt_docs}
        for i, doc in enumerate(ret_docs):
            if doc.id in relevant_id_to_score:  # TODO Related to https://github.com/deepset-ai/haystack/issues/8412
                dcg += relevant_id_to_score[doc.id] / log2(i + 2)  # i + 2 because i is 0-indexed
        return dcg

    @staticmethod
    def calculate_idcg(gt_docs: List[Document]) -> float:
        """
        Calculate the ideal discounted cumulative gain (IDCG) of the ground truth documents.

        :param gt_docs:
            The ground truth documents.
        :returns:
            The ideal discounted cumulative gain (IDCG) of the ground truth documents.
        """
        idcg = 0.0
        for i, doc in enumerate(sorted(gt_docs, key=lambda x: x.score if x.score is not None else 1, reverse=True)):
            # If the document has a score, use it; otherwise, use 1 for binary relevance.
            relevance = doc.score if doc.score is not None else 1
            idcg += relevance / log2(i + 2)  # i + 2 because i is 0-indexed
        return idcg
