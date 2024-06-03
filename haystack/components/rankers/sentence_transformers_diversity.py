# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install \"sentence-transformers>=2.2.0\"'") as torch_and_sentence_transformers_import:
    import torch
    from sentence_transformers import SentenceTransformer


@component
class SentenceTransformersDiversityRanker:
    """
    A Diversity Ranker based on Sentence Transformers.

    Implements a document ranking algorithm that orders documents in such a way as to maximize the overall diversity
    of the documents.

    This component provides functionality to rank a list of documents based on their similarity with respect to the
    query to maximize the overall diversity. It uses a pre-trained Sentence Transformers model to embed the query and
    the Documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import SentenceTransformersDiversityRanker

    ranker = SentenceTransformersDiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine")
    ranker.warm_up()

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        similarity: Literal["dot_product", "cosine"] = "cosine",
        query_prefix: str = "",
        query_suffix: str = "",
        document_prefix: str = "",
        document_suffix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Initialize a SentenceTransformersDiversityRanker.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as `'sentence-transformers/all-MiniLM-L6-v2'`.
        :param top_k: The maximum number of Documents to return per query.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected.
        :param token: The API token used to download private models from Hugging Face.
        :param similarity: Similarity metric for comparing embeddings. Can be set to "dot_product" (default) or
            "cosine".
        :param query_prefix: A string to add to the beginning of the query text before ranking.
            Can be used to prepend the text with an instruction, as required by some embedding models,
            such as E5 and BGE.
        :param query_suffix: A string to add to the end of the query text before ranking.
        :param document_prefix: A string to add to the beginning of each Document text before ranking.
            Can be used to prepend the text with an instruction, as required by some embedding models,
            such as E5 and BGE.
        :param document_suffix: A string to add to the end of each Document text before ranking.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """
        torch_and_sentence_transformers_import.check()

        self.model_name_or_path = model
        if top_k is None or top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.model = None
        if similarity not in ["dot_product", "cosine"]:
            raise ValueError(f"Similarity must be one of 'dot_product' or 'cosine', but got {similarity}.")
        self.similarity = similarity
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_suffix = query_suffix
        self.document_suffix = document_suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.model is None:
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name_or_path,
                device=self.device.to_torch_str(),
                use_auth_token=self.token.resolve_value() if self.token else None,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name_or_path,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            top_k=self.top_k,
            similarity=self.similarity,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            query_suffix=self.query_suffix,
            document_suffix=self.document_suffix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDiversityRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params["device"] is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = (
                self.document_prefix
                + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
                + self.document_suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    def _greedy_diversity_order(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Orders the given list of documents to maximize diversity.

        The algorithm first calculates embeddings for each document and the query. It starts by selecting the document
        that is semantically closest to the query. Then, for each remaining document, it selects the one that, on
        average, is least similar to the already selected documents. This process continues until all documents are
        selected, resulting in a list where each subsequent document contributes the most to the overall diversity of
        the selected set.

        :param query: The search query.
        :param documents: The list of Document objects to be ranked.

        :return: A list of documents ordered to maximize diversity.
        """
        texts_to_embed = self._prepare_texts_to_embed(documents)

        # Calculate embeddings
        doc_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=True)  # type: ignore[attr-defined]
        query_embedding = self.model.encode([self.query_prefix + query + self.query_suffix], convert_to_tensor=True)  # type: ignore[attr-defined]

        # Normalize embeddings to unit length for computing cosine similarity
        if self.similarity == "cosine":
            doc_embeddings /= torch.norm(doc_embeddings, p=2, dim=-1).unsqueeze(-1)
            query_embedding /= torch.norm(query_embedding, p=2, dim=-1).unsqueeze(-1)

        n = len(documents)
        selected: List[int] = []

        # Compute the similarity vector between the query and documents
        query_doc_sim = query_embedding @ doc_embeddings.T

        # Start with the document with the highest similarity to the query
        selected.append(int(torch.argmax(query_doc_sim).item()))

        selected_sum = doc_embeddings[selected[0]] / n

        while len(selected) < n:
            # Compute mean of dot products of all selected documents and all other documents
            similarities = selected_sum @ doc_embeddings.T
            # Mask documents that are already selected
            similarities[selected] = torch.inf
            # Select the document with the lowest total similarity score
            index_unselected = int(torch.argmin(similarities).item())
            selected.append(index_unselected)
            # It's enough just to add to the selected vectors because dot product is distributive
            # It's divided by n for numerical stability
            selected_sum += doc_embeddings[index_unselected] / n

        ranked_docs: List[Document] = [documents[i] for i in selected]

        return ranked_docs

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Rank the documents based on their diversity.

        :param query: The search query.
        :param documents: List of Document objects to be ranker.
        :param top_k: Optional. An integer to override the top_k set during initialization.

        :returns: A dictionary with the following key:
            - `documents`: List of Document objects that have been selected based on the diversity ranking.

        :raises ValueError: If the top_k value is less than or equal to 0.
        :raises RuntimeError: If the component has not been warmed up.
        """
        if self.model is None:
            error_msg = (
                "The component SentenceTransformersDiversityRanker wasn't warmed up. "
                "Run 'warm_up()' before calling 'run()'."
            )
            raise RuntimeError(error_msg)

        if not documents:
            return {"documents": []}

        if top_k is None:
            top_k = self.top_k
        elif top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        diversity_sorted = self._greedy_diversity_order(query=query, documents=documents)

        return {"documents": diversity_sorted[:top_k]}
