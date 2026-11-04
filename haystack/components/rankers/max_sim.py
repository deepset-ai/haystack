# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from haystack import Document, component, logging
from haystack.utils.misc import _deduplicate_documents

logger = logging.getLogger(__name__)


@component
class MaxSimRanker:
    """
    Ranks documents by their similarity to a query using precomputed multivector embeddings.

    It computes a MaxSim score by selecting the highest dot product with a document vector for each query vector,
    then summing these values.

    This component is typically used in two-stage retrieval. Separate dense and multivector embedders prepare
    the document representations, which are saved to a document store. At query time, the corresponding query
    embedders generate both representations. An embedding retriever quickly selects candidates using dense
    similarity. This ranker then uses the multivector embeddings for a more fine-grained comparison of the query
    and each candidate's content.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.rankers import MaxSimRanker

    # Representative multivector embedding for ""How do solar panels work?""
    query_multivector_embedding = [[1.0, 0.0], [0.0, 1.0]]

    documents = [
        Document(content="Wind turbines generate electricity.", multivector_embedding=[[1.0, 0.0], [0.0, 0.2]]),
        Document(content="Solar panels turn sunlight into power.", multivector_embedding=[[0.95, 0.05], [0.15, 0.9]]),
        Document(content="Batteries store energy.", multivector_embedding=[[0.3, 0.1], [0.1, 0.3]]),
        Document(content="Solar cells absorb light.", multivector_embedding=[[0.9, 0.1], [0.1, 0.9]]),
        Document(content="Trees absorb carbon dioxide.", multivector_embedding=[[0.1, 0.0], [0.0, 0.2]]),
        Document(content="Rain falls from clouds.", multivector_embedding=[[0.0, 0.1], [0.1, 0.0]]),
    ]

    ranker = MaxSimRanker(top_k=2)
    result = ranker.run(query_multivector_embedding=query_multivector_embedding, documents=documents)
    for document in result["documents"]:
        print(f"{document.score:.2f}: {document.content}")
    # >> 1.85: Solar panels turn sunlight into power.
    # >> 1.80: Solar cells absorb light.
    ```
    """

    def __init__(self, *, top_k: int = 10) -> None:
        """
        Initialize the MaxSimRanker component.

        :param top_k:
            The maximum number of documents to return.
        :raises ValueError:
            If `top_k` is not greater than 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self, query_multivector_embedding: list[list[float]], documents: list[Document], *, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Returns documents ranked by their MaxSim score for the given query multivector embedding.

        Before ranking, documents without multivector embeddings are skipped with a warning. The remaining
        documents are deduplicated by their id, retaining only the document with the highest existing score
        if a score is present.

        :param query_multivector_embedding:
            The query multivector embedding. All vectors must have the same dimension as the document vectors.
        :param documents:
            The documents to rank. Documents whose `multivector_embedding` is `None` are skipped.
            Present embeddings must be non-empty matrices of finite numbers.
        :param top_k:
            The maximum number of documents to return. Must be greater than 0.
            If `None`, the value provided at initialization is used.
        :returns:
            A dictionary with the following key:
            - `documents`: Up to `top_k` documents sorted by descending MaxSim score,
              with their `score` fields updated.

        :raises ValueError:
            If `top_k` is not greater than 0, a multivector embedding is not a non-empty matrix of finite numbers,
            or query and document vectors have different dimensions.
        """
        top_k = self.top_k if top_k is None else top_k
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if not documents:
            return {"documents": []}

        documents_with_embeddings = []
        document_ids_without_embeddings = []
        for document in documents:
            if document.multivector_embedding is None:
                document_ids_without_embeddings.append(document.id)
            else:
                documents_with_embeddings.append(document)

        if document_ids_without_embeddings:
            logger.warning(
                "Skipping documents without multivector embeddings: {document_ids}",
                document_ids=document_ids_without_embeddings,
            )

        if not documents_with_embeddings:
            return {"documents": []}

        query_vectors = self._prepare_vectors(
            multivector_embedding=query_multivector_embedding, input_name="query_multivector_embedding"
        )
        scored_documents = []
        for document in _deduplicate_documents(documents_with_embeddings):
            document_vectors = self._prepare_vectors(
                multivector_embedding=document.multivector_embedding,
                input_name=f"multivector_embedding of document {document.id!r}",
            )
            if document_vectors.shape[1] != query_vectors.shape[1]:
                raise ValueError(
                    f"multivector_embedding of document {document.id!r} has dimension {document_vectors.shape[1]}, "
                    f"but query_multivector_embedding has dimension {query_vectors.shape[1]}."
                )
            similarities = query_vectors @ document_vectors.T
            score = float(similarities.max(axis=1).sum())
            scored_documents.append((document, score))

        scored_documents.sort(key=lambda item: item[1], reverse=True)
        return {"documents": [replace(document, score=score) for document, score in scored_documents[:top_k]]}

    @staticmethod
    def _prepare_vectors(*, multivector_embedding: list[list[float]] | None, input_name: str) -> NDArray[np.float64]:
        try:
            matrix = np.asarray(multivector_embedding, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{input_name} must be a non-empty matrix of finite numbers.") from error
        if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
            raise ValueError(f"{input_name} must be a non-empty matrix of finite numbers.")
        return matrix
