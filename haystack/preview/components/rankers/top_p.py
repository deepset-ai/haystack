import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
from haystack.lazy_imports import LazyImport
from haystack.preview import Document, component, default_from_dict, default_to_dict


logger = logging.getLogger(__name__)


with LazyImport(
    message="Run 'pip install transformers[torch,sentencepiece]==4.32.1 sentence-transformers>=2.2.0'"
) as torch_and_transformers_import:
    import torch
    from sentence_transformers import CrossEncoder
    from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports


@component
class TopP:
    """
    Ranks documents based on the cumulative probability of the similarity scores between the
    query and the documents using top p sampling.

    Top p sampling selects a subset of the most relevant data points from a larger set of data. The technique
    involves calculating the cumulative probability of the scores of each data point, and then
    selecting the top p percent of data points with the highest cumulative probability.

    In the context of TopPSampler, the `run()` method takes in a query and a set of documents,
    calculates the similarity scores between the query and the documents, and then filters
    the documents based on the cumulative probability of these scores. The TopPSampler provides a
    way to efficiently select the most relevant documents based on their similarity to a given query.

    Usage example:
    ## TODO


    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_p: Optional[float] = 1.0,
        score_field: Optional[str] = "score",
        devices: Optional[List[Union[str, "torch.device"]]] = None,
    ):
        """

        :param model_name_or_path: Path to a pretrained sentence-transformers model.
        :param top_p: Cumulative probability threshold for filtering the documents (usually between 0.9 and 0.99).
        `False` ensures at least one document is returned. If `strict` is set to `True`, then no documents are returned.
        :param score_field: The name of the field that should be used to store the scores a document's meta data.
        :param devices: List of torch devices (for example, cuda:0, cpu, mps) to limit inference to specific devices.
        """
        torch_and_transformers_import.check()
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.top_p = top_p
        self.score_field = score_field
        self.devices, _ = initialize_device_settings(devices=devices)
        self.cross_encoder = CrossEncoder(model_name_or_path, device=str(self.devices[0]))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            top_p=self.top_p,
            score_field=self.score_field,
            devices=self.devices,
            model_name_or_path=self.model_name_or_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopP":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_p: Optional[float] = None):
        """
        Returns a list of documents filtered using `top_p`, based on the similarity scores between the query and the
        documents whose cumulative probability is less than or equal to `top_p`.

        :param query: Query string.
        :param documents: List of Documents.
        :param top_p: Cumulative probability threshold for filtering the documents. If not provided, the top_p value
        set during TopPSampler initialization is used.
        :return: List of Documents sorted by (desc.) similarity with the query.
        """
        if top_p is None:
            top_p = self.top_p if self.top_p else 1.0

        if not documents:
            return []

        # prepare the data for the cross encoder
        query_doc_pairs = [[query, doc.text] for doc in documents]

        # compute the similarity scores for these combinations
        similarity_scores = self.cross_encoder.predict(query_doc_pairs)

        # Apply softmax normalization to the similarity scores
        probs = np.exp(similarity_scores) / np.sum(np.exp(similarity_scores))

        # Sort the probabilities and calculate their cumulative sum
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)

        # Find the indices with cumulative probabilities that exceed top_p
        top_p_indices = np.where(cumulative_probs <= top_p)[0]

        # Map the selected indices back to their original indices
        original_indices = np.argsort(probs)[::-1][top_p_indices]
        # and select the top_p responses
        selected_docs = [documents[i] for i in original_indices]

        # low p resulted in no documents being selected, then
        # return at least one document
        if not selected_docs:
            highest_prob_indices = np.argsort(probs)[::-1]
            selected_docs = [documents[highest_prob_indices[0]]]

        # include prob scores in the results
        if self.score_field:
            for idx, doc in enumerate(selected_docs):
                doc.metadata[self.score_field] = str(sorted_probs[idx])

        return {"documents": selected_docs}
