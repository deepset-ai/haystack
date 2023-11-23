import logging
from typing import List, Optional

from haystack.preview import ComponentError, Document, component
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install \"torch>=1.13\"'") as torch_import:
    import torch


@component
class TopPSampler:
    """
    Filters documents using top-p (nucleus) sampling based on their similarity scores' cumulative probability.

    Usage example:

    ```python
    from haystack.preview import Document
    from haystack.preview.components.samplers import TopPSampler

    sampler = TopPSampler(top_p=0.95)
    docs = [
        Document(text="Berlin", metadata={"similarity_score": -10.6}),
        Document(text="Belgrade", metadata={"similarity_score": -8.9}),
        Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
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

        :param top_p: Cumulative probability threshold (usually between 0.9 and 0.99).
        :param score_field: Field name in a document's metadata containing the scores. Defaults to the Document score
        if not provided.
        """
        torch_import.check()

        self.top_p = top_p
        self.score_field = score_field

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_p: Optional[float] = None):
        """
        Filter documents based on their similarity scores using top-p sampling.

        :param documents: List of Documents to filter.
        :param top_p: Cumulative probability threshold. Defaults to the value set during initialization or 1.0
        if not set.
        :return: List of filtered Documents.
        """
        if not documents:
            return {"documents": []}

        top_p = top_p or self.top_p or 1.0  # default to 1.0 if both are None

        if not 0 <= top_p <= 1:
            raise ComponentError(f"top_p must be between 0 and 1. Got {top_p}.")

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
                "Top-p sampling with p=%s resulted in no documents being selected. "
                "Returning the document with the highest similarity score.",
                top_p,
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
