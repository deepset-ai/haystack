# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np

from haystack import Document, component


@component
class MaxSimRanker:
    """
    Rank precomputed multivector embeddings using the sum of maximum dot products.

    Query and document vectors must come from compatible encoders. No normalization is applied.
    """

    def __init__(self, *, top_k: int = 10) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self, query_embedding: list[list[float]], documents: list[Document], *, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """Score documents using their multi_vector_embedding and return the best matches."""
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        query = np.asarray(query_embedding, dtype=float)
        if query.ndim != 2 or query.size == 0:
            raise ValueError("query_embedding must be a non-empty matrix.")

        scored_documents = []
        for document in documents:
            vectors = np.asarray(document.multi_vector_embedding, dtype=float)
            if vectors.ndim != 2 or vectors.size == 0:
                raise ValueError(f"Document {document.id!r} must have a non-empty multi_vector_embedding matrix.")
            if vectors.shape[1] != query.shape[1]:
                raise ValueError("Query and document vectors must have the same dimension.")
            similarities = query @ vectors.T
            score = float(similarities.max(axis=1).sum())
            scored_documents.append(replace(document, score=score))

        scored_documents.sort(key=lambda document: document.score if document.score is not None else 0.0, reverse=True)
        return {"documents": scored_documents[:limit]}
