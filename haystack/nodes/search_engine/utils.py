from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from haystack import Document
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.search_engine.base import SearchEngine


class SearchEngineSampler(SearchEngine):
    """
    SearchEngineSampler is a SearchEngine decorator providing top_p and top_k sampling of SearchEngine results.
    """

    def __init__(
        self,
        engine: SearchEngine,
        top_p: float = 0.95,
        top_k: int = 5,
        strict_top_k: bool = False,
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        cross_encoder_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.engine = engine
        self.top_p = top_p
        self.top_k = top_k
        self.strict_top_k = strict_top_k
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        self.cross_encoder = CrossEncoder(cross_encoder_model_name_or_path, device=str(self.devices[0]))

    def search(self, query, **kwargs) -> List[Document]:
        top_p = kwargs.pop("top_p", self.top_p)
        docs = self.engine.search(query, **kwargs)
        if docs:
            # prepare the data for the cross encoder
            sts_combinations = [[query, doc.content] for doc in docs]

            # compute the similarity scores for these combinations
            similarity_scores = self.cross_encoder.predict(sts_combinations)

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
            selected_responses = [docs[i] for i in original_indices]

            if not self.strict_top_k:
                selected_responses = selected_responses[: self.top_k]
            else:
                # if we have strict top_k, we need to make sure that we have at least top_k responses
                # if we don't, we need to add more responses from the probs
                if len(selected_responses) < self.top_k:
                    # we need to add more responses
                    sorted_top_k_probs = np.argsort(probs)[::-1][: self.top_k]
                    selected_responses = [docs[i] for i in sorted_top_k_probs]

            # include prob scores in the results
            for idx, doc in enumerate(selected_responses):
                doc.meta["score"] = "{:.2f}".format(sorted_probs[idx])
            return selected_responses
        else:
            return []
