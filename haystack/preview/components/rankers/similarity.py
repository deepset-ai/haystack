import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

from haystack.preview import ComponentError, Document, component, default_from_dict, default_to_dict
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]==4.32.1'") as torch_and_transformers_import:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


@component
class SimilarityRanker:
    """
    Ranks documents based on query similarity.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.rankers import SimilarityRanker

    sampler = SimilarityRanker()
    docs = [Document(text="Paris"), Document(text="Berlin")]
    query = "City in Germany"
    output = sampler.run(query=query, documents=docs)
    docs = output["documents"]
    assert len(docs) == 2
    assert docs[0].text == "Berlin"
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        device: str = "cpu",
    ):
        """
        Creates an instance of SimilarityRanker.

        :param model_name_or_path: Path to a pre-trained sentence-transformers model.
        :param top_k: The maximum number of documents to return per query.
        :param device: torch device (for example, cuda:0, cpu, mps) to limit model inference to a specific device.
        """
        torch_and_transformers_import.check()

        self.model_name_or_path = model_name_or_path
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = device
        self.model = None
        self.tokenizer = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": str(self.model_name_or_path)}

    def warm_up(self):
        """
        Warm up the model and tokenizer used in scoring the documents.
        """
        if self.model_name_or_path and not self.model:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, top_k=self.top_k, device=self.device, model_name_or_path=self.model_name_or_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarityRanker":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Returns a list of documents ranked by their similarity to the given query

        :param query: Query string.
        :param documents: List of Documents.
        :param top_k: The maximum number of documents to return.
        :return: List of Documents sorted by (desc.) similarity with the query.
        """
        if not documents:
            return {"documents": []}

        if top_k is None:
            top_k = self.top_k

        elif top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        # If a model path is provided but the model isn't loaded
        if self.model_name_or_path and not self.model:
            raise ComponentError(
                f"The component {self.__class__.__name__} not warmed up. Run 'warm_up()' before calling 'run()'."
            )

        query_doc_pairs = [[query, doc.text] for doc in documents]
        features = self.tokenizer(
            query_doc_pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(  # type: ignore
            self.device
        )
        with torch.inference_mode():
            similarity_scores = self.model(**features).logits.squeeze()  # type: ignore

        _, sorted_indices = torch.sort(similarity_scores, descending=True)
        ranked_docs = []
        for sorted_index_tensor in sorted_indices:
            i = sorted_index_tensor.item()
            documents[i].score = similarity_scores[i].item()
            ranked_docs.append(documents[i])
        return {"documents": ranked_docs[:top_k]}
