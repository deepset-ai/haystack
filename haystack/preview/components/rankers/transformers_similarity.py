import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

from haystack.preview import ComponentError, Document, component, default_to_dict
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


@component
class TransformersSimilarityRanker:
    """
    Ranks Documents based on their similarity to the query.
    It uses a pre-trained cross-encoder model (from the Hugging Face Hub) to embed the query and the Documents.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.rankers import TransformersSimilarityRanker

    ranker = TransformersSimilarityRanker()
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    assert len(docs) == 2
    assert docs[0].content == "Berlin"
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        token: Union[bool, str, None] = None,
        top_k: int = 10,
    ):
        """
        Creates an instance of TransformersSimilarityRanker.

        :param model_name_or_path: The name or path of a pre-trained cross-encoder model
            from the Hugging Face Hub.
        :param device: The torch device (for example, cuda:0, cpu, mps) to which you want to limit model inference.
        :param token: The API token used to download private models from Hugging Face.
            If this parameter is set to `True`, the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) is used.
        :param top_k: The maximum number of Documents to return per query.
        """
        torch_and_transformers_import.check()

        self.model_name_or_path = model_name_or_path
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = device
        self.token = token
        self.model = None
        self.tokenizer = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": str(self.model_name_or_path)}

    def warm_up(self):
        """
        Warm up the model and tokenizer used for scoring the Documents.
        """
        if self.model_name_or_path and not self.model:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, token=self.token)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=self.token)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            device=self.device,
            model_name_or_path=self.model_name_or_path,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            top_k=self.top_k,
        )

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Returns a list of Documents ranked by their similarity to the given query.

        :param query: Query string.
        :param documents: List of Documents.
        :param top_k: The maximum number of Documents you want the Ranker to return.
        :return: List of Documents sorted by their similarity to the query with the most similar Documents appearing first.
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
                f"The component {self.__class__.__name__} wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        query_doc_pairs = [[query, doc.content] for doc in documents]
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
