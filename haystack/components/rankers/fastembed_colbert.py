# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice

with LazyImport(message="Run 'pip install fastembed'") as fastembed_import:
    import numpy as np
    from fastembed import LateInteractionTextEmbedding


@component
class FastembedColBERTRanker:
    """
    Ranks documents based on their semantic similarity to the query using ColBERT bi-encoder architecture.

    It uses a late-interaction ColBERT model through FastEmbed to embed the query and documents separately,
    then computes relevance scores using the MaxSim algorithm.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.rankers import FastembedColBERTRanker

    ranker = FastembedColBERTRanker()
    docs = [
        Document(content="Paris is the capital of France"),
        Document(content="Berlin is the capital of Germany")
    ]
    query = "What is the capital of Germany?"
    ranker.warm_up()
    result = ranker.run(query=query, documents=docs)
    docs = result["documents"]
    print(docs[0].content)  # "Berlin is the capital of Germany"
    ```
    """

    def __init__(
        self,
        *,
        model: str = "colbert-ir/colbertv2.0",
        device: ComponentDevice | None = None,
        top_k: int = 10,
        query_prefix: str = "",
        query_suffix: str = "",
        document_prefix: str = "",
        document_suffix: str = "",
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        score_threshold: float | None = None,
        batch_size: int = 32,
    ):
        """
        Creates an instance of FastembedColBERTRanker.

        :param model:
            The ColBERT model name. Supported models can be listed with
            `LateInteractionTextEmbedding.list_supported_models()`.
            Default is "colbert-ir/colbertv2.0".
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
        :param top_k:
            The maximum number of documents to return per query.
        :param query_prefix:
            A string to add at the beginning of the query text before ranking.
        :param query_suffix:
            A string to add at the end of the query text before ranking.
        :param document_prefix:
            A string to add at the beginning of each document before ranking.
        :param document_suffix:
            A string to add at the end of each document before ranking.
        :param meta_fields_to_embed:
            List of metadata fields to embed with the document.
        :param embedding_separator:
            Separator to concatenate metadata fields to the document.
        :param score_threshold:
            Use it to return documents with a score above this threshold only.
        :param batch_size:
            The batch size to use for embedding generation. Higher values increase memory usage
            but may improve throughput.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        fastembed_import.check()

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.model = model
        self._embedding_model: "LateInteractionTextEmbedding" | None = None
        self.query_prefix = query_prefix
        self.query_suffix = query_suffix
        self.document_prefix = document_prefix
        self.document_suffix = document_suffix
        self.device = ComponentDevice.resolve_device(device)
        self.top_k = top_k
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.score_threshold = score_threshold
        self.batch_size = batch_size

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def warm_up(self) -> None:
        """
        Initializes the component by loading the ColBERT model.
        """
        if self._embedding_model is None:
            self._embedding_model = LateInteractionTextEmbedding(
                model_name=self.model,
                # FastEmbed uses 'cuda' string for GPU, not torch device format
                device="cuda" if self.device.to_torch_str().startswith("cuda") else "cpu",
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            device=self.device,
            model=self.model,
            top_k=self.top_k,
            query_prefix=self.query_prefix,
            query_suffix=self.query_suffix,
            document_prefix=self.document_prefix,
            document_suffix=self.document_suffix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            score_threshold=self.score_threshold,
            batch_size=self.batch_size,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FastembedColBERTRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def _compute_maxsim_scores(
        self, query_embedding: "np.ndarray", document_embeddings: list["np.ndarray"]
    ) -> list[float]:
        """
        Computes MaxSim scores between query and documents.

        :param query_embedding:
            Query embedding array of shape (num_query_tokens, embedding_dim).
        :param document_embeddings:
            List of document embedding arrays, each of shape (num_doc_tokens, embedding_dim).
        :returns:
            List of relevance scores, one per document.
        """
        scores = []

        for doc_embedding in document_embeddings:
            # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
            similarity_matrix = query_embedding @ doc_embedding.T

            # For each query token, take max similarity across all document tokens
            max_scores_per_query_token = similarity_matrix.max(axis=1)

            total_score = max_scores_per_query_token.sum()
            scores.append(float(total_score))

        return scores

    @component.output_types(documents=list[Document])
    def run(
        self, *, query: str, documents: list[Document], top_k: int | None = None, score_threshold: float | None = None
    ) -> dict[str, list[Document]]:
        """
        Returns list of documents ranked by their similarity to the query using ColBERT.

        :param query:
            The input query to compare the documents to.
        :param documents:
            A list of documents to be ranked.
        :param top_k:
            The maximum number of documents to return.
        :param score_threshold:
            Use it to return documents only with a score above this threshold.
            If set, overrides the value set at initialization.
        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents ranked by relevance to the query,
                          sorted from most similar to least similar.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        if self._embedding_model is None:
            self.warm_up()

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        # Prepare query and documents with prefixes/suffixes
        prepared_query = self.query_prefix + query + self.query_suffix
        prepared_documents = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            prepared_documents.append(
                self.document_prefix
                + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
                + self.document_suffix
            )

        # mypy doesn't know this is set in warm_up
        query_embedding = list(self._embedding_model.query_embed([prepared_query]))[0]  # type: ignore[union-attr]
        document_embeddings = list(self._embedding_model.embed(prepared_documents))  # type: ignore[union-attr]
        scores = self._compute_maxsim_scores(query_embedding, document_embeddings)

        ranked_docs = []
        for idx, score in enumerate(scores):
            document = copy(documents[idx])
            document.score = score
            ranked_docs.append(document)
        ranked_docs.sort(key=lambda x: x.score or 0.0, reverse=True)

        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if (doc.score or 0.0) >= score_threshold]

        return {"documents": ranked_docs[:top_k]}
