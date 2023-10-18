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
    A component that joins input lists of documents from multiple connections and outputs them as one list.

    The component allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
                   The score is only determined by the last node that outputs the document.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    * reciprocal_rank_fusion: combines the documents based on their rank in multiple nodes.

    Example usage in a hybrid retrieval pipeline:
    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    p.add_component(instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever")
    p.add_component(instance=DocumentJoiner(), name="joiner")
    p.connect("bm25_retriever", "joiner")
    p.connect("embedding_retriever", "joiner")
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

        :param join_mode: `concatenate` to combine documents from multiple retrievers, `merge` to aggregate scores of
                          individual documents, `reciprocal_rank_fusion` to apply rank based scoring.
        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                        to each retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k: How many documents should be returned in the output at maximum. By default, all are returned.
        :param sort_by_score: Whether the output list of documents should be sorted by document scores in descending order. By default the output is sorted. Documents without score are handled as if their score was -infinity.
        """
        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(f"DocumentJoiner component does not support '{join_mode}' join_mode.")
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k = top_k
        self.sort_by_score = sort_by_score
        component.set_input_types(self, documents=Variadic[List[Document]])

    @component.output_types(documents=List[Document])
    def run(self, documents):
        """
        Run the DocumentJoiner. This method joins the input lists of documents into one output list based on the join_mode specified during initialization.

        :param documents: An arbitrary number of lists of documents to join.
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
                    "The `DocumentJoiner` component has received some documents with `score=None` - and was requested "
                    "to sort the documents by score, so the `score=None` documents were sorted as if their "
                    "score was `-infinity`."
                )

        if self.top_k:
            output_documents = output_documents[: self.top_k]
        return {"documents": output_documents}

    def _concatenate(self, document_lists):
        """
        Concatenate multiple lists of documents and return only the document with the highest score for duplicate documents.
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
        Merge multiple lists of documents and calculate a weighted sum of the scores of duplicate documents.
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
        Merge multiple lists of documents and assign scores based on reciprocal rank fusion.
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
