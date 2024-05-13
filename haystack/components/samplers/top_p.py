# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from haystack import ComponentError, Document, component, logging
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

    def __init__(self, top_p: float = 1.0, score_field: Optional[str] = None):
        """
        Creates an instance of TopPSampler.

        :param top_p: Float between 0 and 1 representing the cumulative probability threshold for document selection.
            A value of 1.0 indicates no filtering (all documents are retained).
        :param score_field: Name of the field in each document's metadata that contains the score. If None, the default
            document score field is used.
        """
        torch_import.check()

        self.top_p = top_p
        self.score_field = score_field

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_p: Optional[float] = None):
        """
        Filters documents using top-p sampling based on their scores.

        If the specified top_p results in no documents being selected (especially in cases of a low top_p value), the
        method returns the document with the highest similarity score.

        :param documents: List of Document objects to be filtered.
        :param top_p: Optional. A float to override the cumulative probability threshold set during initialization.

        :returns: A dictionary with the following key:
            - `documents`: List of Document objects that have been selected based on the top-p sampling.

        :raises ValueError: If the top_p value is not within the range [0, 1].
        """
        if not documents:
            return {"documents": []}

        top_p = top_p or self.top_p or 1.0  # default to 1.0 if both are None

        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1. Got {top_p}.")

        similarity_scores = torch.tensor(self._collect_scores(documents), dtype=torch.float32)

        # Apply softmax normalization to the similarity scores
        probs = torch.nn.functional.softmax(similarity_scores, dim=-1)

        # Sort the probabilities and calculate their cumulative sum
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Check if the cumulative probabilities are close to top_p with a 1e-6 tolerance
        close_to_top_p = torch.isclose(cumulative_probs, torch.tensor(top_p, device=cumulative_probs.device), atol=1e-6)

        # Combine the close_to_top_p with original condition using logical OR
        condition = (cumulative_probs <= top_p) | close_to_top_p

        # Find the indices with cumulative probabilities that exceed top_p
        top_p_indices = torch.where(torch.BoolTensor(condition))[0]

        # Map the selected indices back to their original indices
        original_indices = sorted_indices[top_p_indices]
        selected_docs = [documents[i.item()] for i in original_indices]

        # If low p resulted in no documents being selected, then
        # return at least one document
        if not selected_docs:
            logger.warning(
                "Top-p sampling with p={top_p} resulted in no documents being selected. "
                "Returning the document with the highest similarity score.",
                top_p=top_p,
            )
            highest_prob_indices = torch.argsort(probs, descending=True)
            selected_docs = [documents[int(highest_prob_indices[0].item())]]

        return {"documents": selected_docs}

    def _collect_scores(self, documents: List[Document]) -> List[float]:
        """
        Collect the scores from the documents' metadata.

        :param documents: List of Documents.
        :return: List of scores.
        """
        if self.score_field:
            missing_scores_docs = [d for d in documents if self.score_field not in d.meta]
            if missing_scores_docs:
                missing_scores_docs_ids = [d.id for d in missing_scores_docs if d.id]
                raise ComponentError(
                    f"Score field '{self.score_field}' not found in metadata of documents "
                    f"with IDs: {missing_scores_docs_ids}."
                    f"Make sure that all documents have a score field '{self.score_field}' in their metadata."
                )
            return [d.meta[self.score_field] for d in documents]
        else:
            missing_scores_docs = [d for d in documents if d.score is None]
            if missing_scores_docs:
                missing_scores_docs_ids = [d.id for d in missing_scores_docs if d.id]
                raise ComponentError(
                    f"Ensure all documents have a valid score value. These docs  {missing_scores_docs_ids} don't."
                )
            return [d.score for d in documents]  # type: ignore ## because Document score is Optional
