# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

with LazyImport(message="Run 'pip install \"sentence-transformers>=4.1.0\"'") as torch_and_sentence_transformers_import:
    import torch
    from sentence_transformers import SentenceTransformer


class DiversityRankingStrategy(Enum):
    """
    The strategy to use for diversity ranking.
    """

    GREEDY_DIVERSITY_ORDER = "greedy_diversity_order"
    MAXIMUM_MARGIN_RELEVANCE = "maximum_margin_relevance"

    def __str__(self) -> str:
        """
        Convert a Strategy enum to a string.
        """
        return self.value

    @staticmethod
    def from_str(string: str) -> "DiversityRankingStrategy":
        """
        Convert a string to a Strategy enum.
        """
        enum_map = {e.value: e for e in DiversityRankingStrategy}
        strategy = enum_map.get(string)
        if strategy is None:
            msg = f"Unknown strategy '{string}'. Supported strategies are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return strategy


class DiversityRankingSimilarity(Enum):
    """
    The similarity metric to use for comparing embeddings.
    """

    DOT_PRODUCT = "dot_product"
    COSINE = "cosine"

    def __str__(self) -> str:
        """
        Convert a Similarity enum to a string.
        """
        return self.value

    @staticmethod
    def from_str(string: str) -> "DiversityRankingSimilarity":
        """
        Convert a string to a Similarity enum.
        """
        enum_map = {e.value: e for e in DiversityRankingSimilarity}
        similarity = enum_map.get(string)
        if similarity is None:
            msg = f"Unknown similarity metric '{string}'. Supported metrics are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return similarity


@component
class SentenceTransformersDiversityRanker:
    """
    A Diversity Ranker based on Sentence Transformers.

    Applies a document ranking algorithm based on one of the two strategies:

    1. Greedy Diversity Order:

        Implements a document ranking algorithm that orders documents in a way that maximizes the overall diversity
        of the documents based on their similarity to the query.

        It uses a pre-trained Sentence Transformers model to embed the query and
        the documents.

    2. Maximum Margin Relevance:

        Implements a document ranking algorithm that orders documents based on their Maximum Margin Relevance (MMR)
        scores.

        MMR scores are calculated for each document based on their relevance to the query and diversity from already
        selected documents. The algorithm iteratively selects documents based on their MMR scores, balancing between
        relevance to the query and diversity from already selected documents. The 'lambda_threshold' controls the
        trade-off between relevance and diversity.

    ### Usage example
    ```python
    from haystack import Document
    from haystack.components.rankers import SentenceTransformersDiversityRanker

    ranker = SentenceTransformersDiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine", strategy="greedy_diversity_order")
    ranker.warm_up()

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """  # noqa: E501

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        similarity: Union[str, DiversityRankingSimilarity] = "cosine",
        query_prefix: str = "",
        query_suffix: str = "",
        document_prefix: str = "",
        document_suffix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        strategy: Union[str, DiversityRankingStrategy] = "greedy_diversity_order",
        lambda_threshold: float = 0.5,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
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
        :param strategy: The strategy to use for diversity ranking. Can be either "greedy_diversity_order" or
                         "maximum_margin_relevance".
        :param lambda_threshold: The trade-off parameter between relevance and diversity. Only used when strategy is
                                 "maximum_margin_relevance".
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        """
        torch_and_sentence_transformers_import.check()

        self.model_name_or_path = model
        if top_k is None or top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.model: Optional[SentenceTransformer] = None
        self.similarity = DiversityRankingSimilarity.from_str(similarity) if isinstance(similarity, str) else similarity
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_suffix = query_suffix
        self.document_suffix = document_suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.strategy = DiversityRankingStrategy.from_str(strategy) if isinstance(strategy, str) else strategy
        self.lambda_threshold = lambda_threshold or 0.5
        self._check_lambda_threshold(self.lambda_threshold, self.strategy)
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.backend = backend

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.model is None:
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name_or_path,
                device=self.device.to_torch_str(),
                token=self.token.resolve_value() if self.token else None,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model_name_or_path,
            top_k=self.top_k,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            similarity=str(self.similarity),
            query_prefix=self.query_prefix,
            query_suffix=self.query_suffix,
            document_prefix=self.document_prefix,
            document_suffix=self.document_suffix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            strategy=str(self.strategy),
            lambda_threshold=self.lambda_threshold,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

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
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(init_params, keys=["token"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
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

        doc_embeddings, query_embedding = self._embed_and_normalize(query, texts_to_embed)

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

    def _embed_and_normalize(self, query, texts_to_embed):
        assert self.model is not None  # verified in run but mypy doesn't see it

        # Calculate embeddings
        doc_embeddings = self.model.encode(texts_to_embed, convert_to_tensor=True)
        query_embedding = self.model.encode([self.query_prefix + query + self.query_suffix], convert_to_tensor=True)

        # Normalize embeddings to unit length for computing cosine similarity
        if self.similarity == DiversityRankingSimilarity.COSINE:
            doc_embeddings /= torch.norm(doc_embeddings, p=2, dim=-1).unsqueeze(-1)
            query_embedding /= torch.norm(query_embedding, p=2, dim=-1).unsqueeze(-1)
        return doc_embeddings, query_embedding

    def _maximum_margin_relevance(
        self, query: str, documents: List[Document], lambda_threshold: float, top_k: int
    ) -> List[Document]:
        """
        Orders the given list of documents according to the Maximum Margin Relevance (MMR) scores.

        MMR scores are calculated for each document based on their relevance to the query and diversity from already
        selected documents.

        The algorithm iteratively selects documents based on their MMR scores, balancing between relevance to the query
        and diversity from already selected documents. The 'lambda_threshold' controls the trade-off between relevance
        and diversity.

        A closer value to 0 favors diversity, while a closer value to 1 favors relevance to the query.

        See : "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"
               https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
        """

        texts_to_embed = self._prepare_texts_to_embed(documents)
        doc_embeddings, query_embedding = self._embed_and_normalize(query, texts_to_embed)
        top_k = top_k if top_k else len(documents)

        selected: List[int] = []
        query_similarities_as_tensor = query_embedding @ doc_embeddings.T
        query_similarities = query_similarities_as_tensor.reshape(-1)
        idx = int(torch.argmax(query_similarities))
        selected.append(idx)
        while len(selected) < top_k:
            best_idx = None
            best_score = -float("inf")
            for idx, _ in enumerate(documents):
                if idx in selected:
                    continue
                relevance_score = query_similarities[idx]
                diversity_score = max(
                    doc_embeddings[idx] @ doc_embeddings[j].permute(*torch.arange(doc_embeddings[j].ndim - 1, -1, -1))
                    for j in selected
                )
                mmr_score = lambda_threshold * relevance_score - (1 - lambda_threshold) * diversity_score
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                raise ValueError("No best document found, check if the documents list contains any documents.")
            selected.append(best_idx)

        return [documents[i] for i in selected]

    @staticmethod
    def _check_lambda_threshold(lambda_threshold: float, strategy: DiversityRankingStrategy):
        if (strategy == DiversityRankingStrategy.MAXIMUM_MARGIN_RELEVANCE) and not 0 <= lambda_threshold <= 1:
            raise ValueError(f"lambda_threshold must be between 0 and 1, but got {lambda_threshold}.")

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        lambda_threshold: Optional[float] = None,
    ) -> Dict[str, List[Document]]:
        """
        Rank the documents based on their diversity.

        :param query: The search query.
        :param documents: List of Document objects to be ranker.
        :param top_k: Optional. An integer to override the top_k set during initialization.
        :param lambda_threshold: Override the trade-off parameter between relevance and diversity. Only used when
                                strategy is "maximum_margin_relevance".

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
        elif not 0 < top_k <= len(documents):
            raise ValueError(f"top_k must be between 1 and {len(documents)}, but got {top_k}")

        if self.strategy == DiversityRankingStrategy.MAXIMUM_MARGIN_RELEVANCE:
            if lambda_threshold is None:
                lambda_threshold = self.lambda_threshold
            self._check_lambda_threshold(lambda_threshold, self.strategy)
            re_ranked_docs = self._maximum_margin_relevance(
                query=query, documents=documents, lambda_threshold=lambda_threshold, top_k=top_k
            )
        else:
            re_ranked_docs = self._greedy_diversity_order(query=query, documents=documents)

        return {"documents": re_ranked_docs[:top_k]}
