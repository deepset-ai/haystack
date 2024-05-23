# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import defaultdict
from math import inf
from typing import List, Optional

from haystack import Document, component, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


@component
class DocumentJoiner:
    """
    A component that joins multiple list of Documents into a single list.

    It supports different joins modes:
    - concatenate: Keeps the highest scored Document in case of duplicates.
    - merge: Merge a calculate a weighted sum of the scores of duplicate Documents.
    - reciprocal_rank_fusion: Merge and assign scores based on reciprocal rank fusion.

    Usage example:
    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    p.add_component(
            instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
            name="text_embedder",
        )
    p.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever")
    p.add_component(instance=DocumentJoiner(), name="joiner")
    p.connect("bm25_retriever", "joiner")
    p.connect("embedding_retriever", "joiner")
    p.connect("text_embedder", "embedding_retriever")
    query = "What is the capital of France?"
    p.run(data={"query": query})
    ```
    """

    def __init__(
        self,
        join_mode: str = "concatenate",
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        sort_by_score: bool = True,
    ):
        """
        Create an DocumentJoiner component.

        :param join_mode:
            Specifies the join mode to use. Available modes:
            - `concatenate`
            - `merge`
            - `reciprocal_rank_fusion`
        :param weights:
            Weight for each list of Documents received, must have the same length as the number of inputs.
            If `join_mode` is `concatenate` this parameter is ignored.
        :param top_k:
            The maximum number of Documents to return.
        :param sort_by_score:
            If True sorts the Documents by score in descending order.
            If a Document has no score, it is handled as if its score is -infinity.
        """
        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(f"DocumentJoiner component does not support '{join_mode}' join_mode.")
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=List[Document])
    def run(self, documents: Variadic[List[Document]], top_k: Optional[int] = None):
        """
        Joins multiple lists of Documents into a single list depending on the `join_mode` parameter.

        :param documents:
            List of list of Documents to be merged.
        :param top_k:
            The maximum number of Documents to return. Overrides the instance's `top_k` if provided.

        :returns:
            A dictionary with the following keys:
            - `documents`: Merged list of Documents
        """
        output_documents = []
        if self.join_mode == "concatenate":
            output_documents = self._concatenate(documents)
        elif self.join_mode == "merge":
            output_documents = self._merge(documents)
        elif self.join_mode == "reciprocal_rank_fusion":
            output_documents = self._reciprocal_rank_fusion(documents)

        if self.sort_by_score:
            output_documents = sorted(
                output_documents, key=lambda doc: doc.score if doc.score is not None else -inf, reverse=True
            )
            if any(doc.score is None for doc in output_documents):
                logger.info(
                    "Some of the Documents DocumentJoiner got have score=None. It was configured to sort Documents by "
                    "score, so those with score=None were sorted as if they had a score of -infinity."
                )

        if top_k:
            output_documents = output_documents[:top_k]
        elif self.top_k:
            output_documents = output_documents[: self.top_k]

        return {"documents": output_documents}

    def _concatenate(self, document_lists):
        """
        Concatenate multiple lists of Documents and return only the Document with the highest score for duplicate Documents.
        """
        output = []
        docs_per_id = defaultdict(list)
        for doc in itertools.chain.from_iterable(document_lists):
            docs_per_id[doc.id].append(doc)
        for docs in docs_per_id.values():
            doc_with_best_score = max(docs, key=lambda doc: doc.score if doc.score else -inf)
            output.append(doc_with_best_score)
        return output

    def _merge(self, document_lists):
        """
        Merge multiple lists of Documents and calculate a weighted sum of the scores of duplicate Documents.
        """
        scores_map = defaultdict(int)
        documents_map = {}
        weights = self.weights if self.weights else [1 / len(document_lists)] * len(document_lists)

        for documents, weight in zip(document_lists, weights):
            for doc in documents:
                scores_map[doc.id] += (doc.score if doc.score else 0) * weight
                documents_map[doc.id] = doc

        for doc in documents_map.values():
            doc.score = scores_map[doc.id]

        return documents_map.values()

    def _reciprocal_rank_fusion(self, document_lists):
        """
        Merge multiple lists of Documents and assign scores based on reciprocal rank fusion.

        The constant k is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        k = 61

        scores_map = defaultdict(int)
        documents_map = {}
        weights = self.weights if self.weights else [1 / len(document_lists)] * len(document_lists)

        # Calculate weighted reciprocal rank fusion score
        for documents, weight in zip(document_lists, weights):
            for rank, doc in enumerate(documents):
                scores_map[doc.id] += (weight * len(document_lists)) / (k + rank)
                documents_map[doc.id] = doc

        # Normalize scores. Note: len(results) / k is the maximum possible score,
        # achieved by being ranked first in all doc lists with non-zero weight.
        for _id in scores_map:
            scores_map[_id] /= len(document_lists) / k

        for doc in documents_map.values():
            doc.score = scores_map[doc.id]

        return documents_map.values()
