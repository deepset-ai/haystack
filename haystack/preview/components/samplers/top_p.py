import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from haystack.preview import ComponentError

from haystack.lazy_imports import LazyImport
from haystack.preview import Document, component, default_from_dict, default_to_dict


logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]==4.32.1'") as torch_and_transformers_import:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


@component
class TopPSampler:
    """
    Selects documents based on the cumulative probability of the similarity scores between the
    query and the documents using top p sampling.

    Top p sampling selects a subset of the most relevant data points from a larger set of data. The technique
    involves calculating the cumulative probability of the scores of each data point, and then
    selecting the top p percent of data points with the highest cumulative probability.

    In the context of TopPSampler, the `run()` method takes in a query and a set of documents,
    calculates the similarity scores between the query and the documents, and then filters
    the documents based on the cumulative probability of these scores. The TopPSampler provides a
    way to efficiently select the most relevant documents based on their similarity to a given query.

    Usage example:
    ```
    from haystack.preview import Document
    from haystack.preview.components.samplers import TopPSampler

    sampler = TopPSampler(top_p=0.95)
    docs = [Document(text="Paris"), Document(text="Berlin")]
    query = "City in Germany"
    output = sampler.run(query=query, documents=docs)
    docs = output["documents"]
    assert len(docs) == 1
    assert docs[0].text == "Berlin"
    ```
    """

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        top_p: Optional[float] = 1.0,
        score_field: Optional[str] = "similarity_score",
        device: Optional[str] = "cpu",
    ):
        """
        Creates an instance of TopPSampler.

        :param model_name_or_path: Path to a pre-trained sentence-transformers model. If not specified, no document
        scoring will be performed, and it is assumed that documents already have scores in the metadata specified under
        the `score_field` key.
        :param top_p: Cumulative probability threshold for filtering the documents (usually between 0.9 and 0.99).
        `False` ensures at least one document is returned. If `strict` is set to `True`, then no documents are returned.
        :param score_field: The name of the field that should be used to store the scores in a document's metadata.
        :param device: torch device (for example, cuda:0, cpu, mps) to limit inference to a specific device.
        """
        torch_and_transformers_import.check()
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.top_p = top_p
        self.score_field = score_field
        self.device = device
        self.model = None
        self.tokenizer = None

    def warm_up(self):
        if self.model_name_or_path and (self.model is None and self.tokenizer is None):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            top_p=self.top_p,
            score_field=self.score_field,
            device=self.device,
            model_name_or_path=self.model_name_or_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopPSampler":
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

        if self.model_name_or_path and self.model is None:
            raise ComponentError(
                f"The component {self.__class__.__name__} not warmed up. Run 'warm_up()' before calling 'run()'."
            )
        if not self.model_name_or_path and not self.has_scores(documents):
            raise ComponentError(
                f"The component {self.__class__.__name__} requires scores in the documents' metadata. "
                f"Either set 'model_name_or_path' to scored documents or add scores to the documents."
            )

        needs_scoring = self.model_name_or_path and not self.has_scores(documents)
        if needs_scoring:
            # prepare the data for the model
            query_doc_pairs = [[query, doc.text] for doc in documents]
            features = self.tokenizer(query_doc_pairs, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                similarity_scores = self.model(**features).logits.squeeze()
        else:
            similarity_scores = torch.tensor(self.collect_scores(documents), dtype=torch.float32)

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
            selected_docs = [documents[highest_prob_indices[0].item()]]

        # Include prob scores in the results
        if self.score_field and needs_scoring:
            for idx, doc in enumerate(selected_docs):
                doc.metadata[self.score_field] = str(sorted_probs[idx].item())

        return {"documents": selected_docs}

    def collect_scores(self, documents: List[Document]) -> List[float]:
        """
        Collect the scores from the documents' metadata.
        :param documents: List of Documents.
        :return: List of scores.
        """
        return [d.metadata[self.score_field] for d in documents]

    def has_scores(self, documents: List[Document]) -> bool:
        """
        Check if the documents have scores in their metadata.
        :param documents: List of Documents.
        :return: True if the documents have scores in their metadata, False otherwise.
        """
        return all(self.score_field in d.metadata for d in documents)
