# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install \"torch>=1.13\"'") as torch_import:
    import torch


@component
class TopPSampler:
    """
    Implements top-p (nucleus) sampling for document filtering based on cumulative probability scores.

    This component provides functionality to filter a list of documents by selecting those whose scores fall
    within the top 'p' percent of the cumulative distribution. It is useful for focusing on high-probability
    documents while filtering out less relevant ones based on their assigned scores.

    Usage example:

    ```python
    from haystack import Document
    from haystack.components.samplers import TopPSampler

    sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
    docs = [
        Document(content="Berlin", meta={"similarity_score": -10.6}),
        Document(content="Belgrade", meta={"similarity_score": -8.9}),
        Document(content="Sarajevo", meta={"similarity_score": -4.6}),
    ]
    output = sampler.run(documents=docs)
    docs = output["documents"]
    assert len(docs) == 1
    assert docs[0].content == "Sarajevo"
    ```
    """

    def __init__(self, top_p: float = 1.0, score_field: Optional[str] = None, min_top_k: Optional[int] = None):
        """
        Creates an instance of TopPSampler.

        :param top_p: Float between 0 and 1 representing the cumulative probability threshold for document selection.
            A value of 1.0 indicates no filtering (all documents are retained).
        :param score_field: Name of the field in each document's metadata that contains the score. If None, the default
            document score field is used.
        :param min_top_k: If specified, the minimum number of documents to return. If the top_p selects
            fewer documents, additional ones with the next highest scores are added to the selection.
        """
        torch_import.check()

        self.top_p = top_p
        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1. Got {top_p}.")
        self.score_field = score_field
        self.min_top_k = min_top_k

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_p: Optional[float] = None):
        """
        Filters documents using top-p sampling based on their scores.

        If the specified top_p results in no documents being selected (especially in cases of a low top_p value), the
        method returns the document with the highest score.

        :param documents: List of Document objects to be filtered.
        :param top_p: If specified, a float to override the cumulative probability threshold set during initialization.

        :returns: A dictionary with the following key:
            - `documents`: List of Document objects that have been selected based on the top-p sampling.
        :raises ValueError: If the top_p value is not within the range [0, 1].
        """
        if not documents:
            return {"documents": []}

        top_p = top_p or self.top_p
        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1. Got {top_p}.")

        documents_with_scores, scores = self._get_documents_and_scores(documents)
        if len(documents_with_scores) == 0:
            logger.warning("No documents with scores found. Returning the original documents.")
            return {"documents": documents}

        sorted_docs_with_scores = sorted(zip(documents_with_scores, scores), key=lambda x: x[1], reverse=True)
        sorted_documents, sorted_scores = [list(t) for t in zip(*sorted_docs_with_scores)]

        tensor_scores = torch.tensor(sorted_scores, dtype=torch.float32)
        probs = torch.nn.functional.softmax(tensor_scores, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Check if the cumulative probabilities are close to top_p with a 1e-6 tolerance
        close_to_top_p = torch.isclose(cumulative_probs, torch.tensor(top_p, device=cumulative_probs.device), atol=1e-6)

        # Combine the close_to_top_p with original condition using logical OR
        condition = (cumulative_probs <= top_p) | close_to_top_p

        # Find the indices with cumulative probabilities that exceed top_p
        top_p_indices = torch.where(torch.BoolTensor(condition))[0]

        # Map the selected indices back to their original indices
        selected_docs = [sorted_documents[i.item()] for i in top_p_indices]

        if self.min_top_k and len(selected_docs) < self.min_top_k:
            selected_docs = sorted_documents[: self.min_top_k]

        # If low p resulted in no documents being selected, then return at least one document
        if len(selected_docs) == 0:
            logger.warning(
                "Top-p sampling with p={top_p} resulted in no documents being selected. "
                "Returning the document with the highest score.",
                top_p=top_p,
            )
            selected_docs = [sorted_documents[0]]

        return {"documents": selected_docs}

    @staticmethod
    def _get_doc_score(doc: Document, score_field: Optional[str] = None) -> Optional[float]:
        """
        Get the score of a document.

        :param doc: Document object.
        :param score_field: Name of the field in the document's metadata that contains the score.
            If None, the document score field is used.

        :return: Score of the document.
        """
        if score_field:
            score = doc.meta.get(score_field)
        else:
            score = doc.score

        if not isinstance(score, float):
            score = None
        return score

    def _get_documents_and_scores(self, documents: List[Document]) -> Tuple[List[Document], List[float]]:
        """
        Checks if documents have scores in their metadata or score field and returns the documents with scores.

        :param documents: List of Documents.
        :return: List of scores.
        """
        docs_with_scores = []
        scores = []
        docs_missing_scores = []
        for doc in documents:
            score = self._get_doc_score(doc=doc, score_field=self.score_field)
            if score is None:
                docs_missing_scores.append(doc)
            else:
                scores.append(score)
                docs_with_scores.append(doc)

        if len(docs_missing_scores) > 0:
            missing_scores_docs_ids = [d.id for d in docs_missing_scores if d.id]
            if self.score_field:
                logger.warning(
                    "Score field '{score_field}' not found in metadata of documents with IDs: {doc_ids}."
                    "Make sure that all documents have a score field '{score_field_2}' in their metadata.",
                    score_field=self.score_field,
                    doc_ids=",".join(missing_scores_docs_ids),
                    score_field_2=self.score_field,
                )
            else:
                logger.warning(
                    "Ensure all documents have a valid score value. These documents {doc_ids} are missing scores.",
                    doc_ids=",".join(missing_scores_docs_ids),
                )
        return docs_with_scores, scores
