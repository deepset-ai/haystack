import itertools
import logging
from collections import defaultdict
from math import inf
from typing import List, Optional
from canals.component.types import Variadic

from haystack.preview import component, Document


logger = logging.getLogger(__name__)


@component
class DocumentJoiner:
    """
    A component that joins input lists of Documents from multiple connections and outputs them as one list.

    The component allows multiple join modes:
    * concatenate: Combine Documents from multiple components. Discards duplicate Documents.
                    Documents get their scores from the last component in the pipeline that assigns scores.
                    This join mode doesn't influence Document scores.
    * merge: Merge scores of duplicate Documents coming from multiple components.
            Optionally, you can assign a weight to the scores and set the top_k limit for this join mode.
            You can also use this join mode to rerank retrieved Documents.
    * reciprocal_rank_fusion: Combine Documents into a single list based on their ranking received from multiple components.

    Example usage in a hybrid retrieval pipeline:
    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    p.add_component(
            instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
            name="text_embedder",
        )
    p.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever")
    p.add_component(instance=DocumentJoiner(), name="joiner")
    p.connect("bm25_retriever", "joiner")
    p.connect("embedding_retriever", "joiner")
    p.connect("text_embedder", "embedding_retriever")
    query = "What is the capital of France?"
    p.run(data={"bm25_retriever": {"query": query},
                "text_embedder": {"text": query}})
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
        Initialize the DocumentJoiner.

        :param join_mode: Specifies the join mode to use. Available modes: `concatenate` to combine Documents from multiple Retrievers, `merge` to aggregate the scores of
                          individual Documents, `reciprocal_rank_fusion` to apply rank-based scoring.
        :param weights: A component-wise list (the length of the list must be equal to the number of input components) of weights for
                        adjusting Document scores when using the `merge` join_mode. By default, equal weight is given
                        to each Retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k: The maximum number of Documents to be returned as output. By default, returns all Documents.
        :param sort_by_score: Whether the output list of Documents should be sorted by Document scores in descending order.
                              By default, the output is sorted.
                              Documents without score are handled as if their score was -infinity.
        """
        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(f"DocumentJoiner component does not support '{join_mode}' join_mode.")
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=List[Document])
    def run(self, documents: Variadic[List[Document]]):
        """
        Run the DocumentJoiner. This method joins the input lists of Documents into one output list based on the join_mode specified during initialization.

        :param documents: An arbitrary number of lists of Documents to join.
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

        if self.top_k:
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
        for documents in document_lists:
            for rank, doc in enumerate(documents):
                scores_map[doc.id] += 1 / (k + rank)
                documents_map[doc.id] = doc

        for doc in documents_map.values():
            doc.score = scores_map[doc.id]

        return documents_map.values()
