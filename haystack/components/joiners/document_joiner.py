# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import defaultdict
from enum import Enum
from math import inf
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


class JoinMode(Enum):
    """
    Enum for join mode.
    """

    CONCATENATE = "concatenate"
    MERGE = "merge"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
    DISTRIBUTION_BASED_RANK_FUSION = "distribution_based_rank_fusion"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "JoinMode":
        """
        Convert a string to a JoinMode enum.
        """
        enum_map = {e.value: e for e in JoinMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown join mode '{string}'. Supported modes in DocumentJoiner are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class DocumentJoiner:
    """
    Joins multiple lists of documents into a single list.

    It supports different join modes:
    - concatenate: Keeps the highest-scored document in case of duplicates.
    - merge: Calculates a weighted sum of scores for duplicates and merges them.
    - reciprocal_rank_fusion: Merges and assigns scores based on reciprocal rank fusion.
    - distribution_based_rank_fusion: Merges and assigns scores based on scores distribution in each Retriever.

    ### Usage example:

    ```python
    from haystack import Pipeline, Document
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    document_store = InMemoryDocumentStore()
    docs = [Document(content="Paris"), Document(content="Berlin"), Document(content="London")]
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    embedder.warm_up()
    docs_embeddings = embedder.run(docs)
    document_store.write_documents(docs_embeddings['documents'])

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
    p.run(data={"query": query, "text": query, "top_k": 1})
    ```
    """

    def __init__(
        self,
        join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        sort_by_score: bool = True,
    ):
        """
        Creates a DocumentJoiner component.

        :param join_mode:
            Specifies the join mode to use. Available modes:
            - `concatenate`: Keeps the highest-scored document in case of duplicates.
            - `merge`: Calculates a weighted sum of scores for duplicates and merges them.
            - `reciprocal_rank_fusion`: Merges and assigns scores based on reciprocal rank fusion.
            - `distribution_based_rank_fusion`: Merges and assigns scores based on scores
            distribution in each Retriever.
        :param weights:
            Assign importance to each list of documents to influence how they're joined.
            This parameter is ignored for
            `concatenate` or `distribution_based_rank_fusion` join modes.
            Weight for each list of documents must match the number of inputs.
        :param top_k:
            The maximum number of documents to return.
        :param sort_by_score:
            If `True`, sorts the documents by score in descending order.
            If a document has no score, it is handled as if its score is -infinity.
        """
        if isinstance(join_mode, str):
            join_mode = JoinMode.from_str(join_mode)
        join_mode_functions = {
            JoinMode.CONCATENATE: DocumentJoiner._concatenate,
            JoinMode.MERGE: self._merge,
            JoinMode.RECIPROCAL_RANK_FUSION: self._reciprocal_rank_fusion,
            JoinMode.DISTRIBUTION_BASED_RANK_FUSION: DocumentJoiner._distribution_based_rank_fusion,
        }
        self.join_mode_function = join_mode_functions[join_mode]
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=List[Document])
    def run(self, documents: Variadic[List[Document]], top_k: Optional[int] = None):
        """
        Joins multiple lists of Documents into a single list depending on the `join_mode` parameter.

        :param documents:
            List of list of documents to be merged.
        :param top_k:
            The maximum number of documents to return. Overrides the instance's `top_k` if provided.

        :returns:
            A dictionary with the following keys:
            - `documents`: Merged list of Documents
        """
        output_documents = []

        documents = list(documents)
        output_documents = self.join_mode_function(documents)

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

    @staticmethod
    def _concatenate(document_lists: List[List[Document]]) -> List[Document]:
        """
        Concatenate multiple lists of Documents and return only the Document with the highest score for duplicates.
        """
        output = []
        docs_per_id = defaultdict(list)
        for doc in itertools.chain.from_iterable(document_lists):
            docs_per_id[doc.id].append(doc)
        for docs in docs_per_id.values():
            doc_with_best_score = max(docs, key=lambda doc: doc.score if doc.score else -inf)
            output.append(doc_with_best_score)
        return output

    def _merge(self, document_lists: List[List[Document]]) -> List[Document]:
        """
        Merge multiple lists of Documents and calculate a weighted sum of the scores of duplicate Documents.
        """
        # This check prevents a division by zero when no documents are passed
        if not document_lists:
            return []

        scores_map: dict = defaultdict(int)
        documents_map = {}
        weights = self.weights if self.weights else [1 / len(document_lists)] * len(document_lists)

        for documents, weight in zip(document_lists, weights):
            for doc in documents:
                scores_map[doc.id] += (doc.score if doc.score else 0) * weight
                documents_map[doc.id] = doc

        for doc in documents_map.values():
            doc.score = scores_map[doc.id]

        return list(documents_map.values())

    def _reciprocal_rank_fusion(self, document_lists: List[List[Document]]) -> List[Document]:
        """
        Merge multiple lists of Documents and assign scores based on reciprocal rank fusion.

        The constant k is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        # This check prevents a division by zero when no documents are passed
        if not document_lists:
            return []

        k = 61

        scores_map: dict = defaultdict(int)
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

        return list(documents_map.values())

    @staticmethod
    def _distribution_based_rank_fusion(document_lists: List[List[Document]]) -> List[Document]:
        """
        Merge multiple lists of Documents and assign scores based on Distribution-Based Score Fusion.

        (https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18)
        If a Document is in more than one retriever, the one with the highest score is used.
        """
        for documents in document_lists:
            if len(documents) == 0:
                continue

            scores_list = []

            for doc in documents:
                scores_list.append(doc.score if doc.score is not None else 0)

            mean_score = sum(scores_list) / len(scores_list)
            std_dev = (sum((x - mean_score) ** 2 for x in scores_list) / len(scores_list)) ** 0.5
            min_score = mean_score - 3 * std_dev
            max_score = mean_score + 3 * std_dev
            delta_score = max_score - min_score

            for doc in documents:
                doc.score = (doc.score - min_score) / delta_score if delta_score != 0.0 else 0.0
                # if all docs have the same score delta_score is 0, the docs are uninformative for the query

        output = DocumentJoiner._concatenate(document_lists=document_lists)

        return output

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            join_mode=str(self.join_mode),
            weights=self.weights,
            top_k=self.top_k,
            sort_by_score=self.sort_by_score,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentJoiner":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
