import logging
from typing import List, Optional, Dict, Any

from haystack.preview import ComponentError, Document, component, default_from_dict, default_to_dict
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]==4.32.1'") as torch_and_transformers_import:
    import torch


@component
class TopPSampler:
    """
    Selects documents based on the cumulative probability of the Document similarity scores using top p (nucleus)
    sampling.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.samplers import TopPSampler

    sampler = TopPSampler(top_p=0.95)
    docs = [
        Document(text="Berlin", metadata={"similarity_score": -10.6}),
        Document(text="Belgrade", metadata={"similarity_score": -8.9}),
        Document(text="Sarajevo", metadata={"similarity_score": -4.6}),
    ]
    query = "City in Bosnia and Herzegovina"
    output = sampler.run(documents=docs)
    docs = output["documents"]
    assert len(docs) == 1
    assert docs[0].text == "Sarajevo"
    ```
    """

    def __init__(self, top_p: float = 1.0, score_field: Optional[str] = None):
        """
        Creates an instance of TopPSampler.

        :param top_p: Cumulative probability threshold for filtering the documents (usually between 0.9 and 0.99).
        :param score_field: The name of the field that should be used to resolve the scores in a document's metadata.
        If no score field is provided (default), the component will assume that the scores are stored in the Document
        score field.
        """
        torch_and_transformers_import.check()

        self.top_p = top_p
        self.score_field = score_field

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, top_p=self.top_p, score_field=self.score_field)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopPSampler":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], top_p: Optional[float] = None):
        """
        Returns a list of documents filtered using `top_p`, based on the similarity scores of the documents whose
        cumulative probability is less than or equal to `top_p`.

        :param documents: List of Documents.
        :param top_p: Cumulative probability threshold for filtering the documents. If top_p is not provided at
        initialization then top_p will default to 1.0
        :return: List of Documents whose cumulative probability is less than or equal to `top_p`.
        """
        if not documents:
            return {"documents": []}

        top_p = top_p or self.top_p or 1.0  # default to 1.0 if both are None

        if not 0 <= top_p <= 1:
            raise ComponentError(f"top_p must be between 0 and 1. Got {top_p}.")

        similarity_scores = torch.tensor(self._collect_scores(documents), dtype=torch.float32)

        # Apply softmax normalization to the similarity scores
        probs = torch.exp(similarity_scores) / torch.sum(torch.exp(similarity_scores))

        # Sort the probabilities and calculate their cumulative sum
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find the indices with cumulative probabilities that exceed top_p
        top_p_indices = torch.where(cumulative_probs <= top_p)[0]

        # Map the selected indices back to their original indices
        original_indices = sorted_indices[top_p_indices]
        selected_docs = [documents[i.item()] for i in original_indices]

        # If low p resulted in no documents being selected, then
        # return at least one document
        if not selected_docs:
            highest_prob_indices = torch.argsort(probs, descending=True)
            selected_docs = [documents[int(highest_prob_indices[0].item())]]

        # Include prob scores in the results
        if self.score_field:
            for idx, doc in enumerate(selected_docs):
                doc.metadata[self.score_field] = str(sorted_probs[idx].item())

        return {"documents": selected_docs}

    def _collect_scores(self, documents: List[Document]) -> List[float]:
        """
        Collect the scores from the documents' metadata.
        :param documents: List of Documents.
        :return: List of scores.
        """
        if self.score_field:
            have_scores_in_metadata = all(self.score_field in d.metadata for d in documents)
            if not have_scores_in_metadata:
                raise ComponentError(
                    f"Score field '{self.score_field}' not found in metadata of all documents. "
                    f"Make sure that all documents have a score field '{self.score_field}' in their metadata."
                )
            return [d.metadata[self.score_field] for d in documents]
        else:
            # If no score field is provided, assume the scores are stored in the score
            have_scores = all(d.score for d in documents)
            if not have_scores:
                raise ComponentError(
                    "At least one Document score is None. Make sure all documents have a valid score value."
                )
            return [d.score for d in documents]  # type: ignore ## because Document score is Optional
