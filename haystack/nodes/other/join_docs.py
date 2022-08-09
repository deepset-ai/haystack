from collections import defaultdict
import logging
from math import inf

from typing import Optional, List

from haystack.schema import Document
from haystack.nodes.other.join import JoinNode

logger = logging.getLogger(__name__)


class JoinDocuments(JoinNode):
    """
    A node to join documents outputted by multiple retriever nodes.

    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
                   The score is only determined by the last node that outputs the document.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    * reciprocal_rank_fusion: combines the documents based on their rank in multiple nodes.
    """

    outgoing_edges = 1

    def __init__(
        self,
        join_mode: str = "concatenate",
        weights: Optional[List[float]] = None,
        top_k_join: Optional[int] = None,
        sort_by_score: bool = True,
    ):
        """
        :param join_mode: `concatenate` to combine documents from multiple retrievers `merge` to aggregate scores of
                          individual documents, `reciprocal_rank_fusion` to apply rank based scoring.
        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                        to each retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k_join: Limit documents to top_k based on the resulting scores of the join.
        :param sort_by_score: Whether to sort the incoming documents by their score. Set this to True if all your
                              Documents are coming with `score` values. Set to False if any of the Documents come
                              from sources where the `score` is set to `None`, like `TfidfRetriever` on Elasticsearch.
        """
        assert join_mode in [
            "concatenate",
            "merge",
            "reciprocal_rank_fusion",
        ], f"JoinDocuments node does not support '{join_mode}' join_mode."

        assert not (
            weights is not None and join_mode == "concatenate"
        ), "Weights are not compatible with 'concatenate' join_mode."

        super().__init__()

        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join
        self.sort_by_score = sort_by_score

    def run_accumulated(self, inputs: List[dict], top_k_join: Optional[int] = None):  # type: ignore
        results = [inp["documents"] for inp in inputs]
        document_map = {doc.id: doc for result in results for doc in result}

        if self.join_mode == "concatenate":
            scores_map = self._concatenate_results(results)
        elif self.join_mode == "merge":
            scores_map = self._calculate_comb_sum(results)
        elif self.join_mode == "reciprocal_rank_fusion":
            scores_map = self._calculate_rrf(results)
        else:
            raise ValueError(f"Invalid join_mode: {self.join_mode}")

        # only sort the docs if that was requested
        if self.sort_by_score:
            sorted_docs = sorted(scores_map.items(), key=lambda d: d[1] if d[1] is not None else -inf, reverse=True)
            if any(s is None for s in scores_map.values()):
                logger.info(
                    "The `JoinDocuments` node has received some documents with `score=None` - and was requested "
                    "to sort the documents by score, so the `score=None` documents got sorted as if their "
                    "score would be `-infinity`."
                )
        else:
            sorted_docs = [(k, v) for k, v in scores_map.items()]

        if not top_k_join:
            top_k_join = self.top_k_join
        if not top_k_join:
            top_k_join = len(sorted_docs)

        docs = []
        for (id, score) in sorted_docs[:top_k_join]:
            doc = document_map[id]
            doc.score = score
            docs.append(doc)

        output = {"documents": docs, "labels": inputs[0].get("labels", None)}

        return output, "output_1"

    def run_batch_accumulated(self, inputs: List[dict], top_k_join: Optional[int] = None):  # type: ignore
        # Join single document lists
        if isinstance(inputs[0]["documents"][0], Document):
            return self.run(inputs=inputs, top_k_join=top_k_join)
        # Join lists of document lists
        else:
            output_docs = []
            incoming_edges = [inp["documents"] for inp in inputs]
            for idx in range(len(incoming_edges[0])):
                cur_docs_to_join = []
                for edge in incoming_edges:
                    cur_docs_to_join.append({"documents": edge[idx]})
                cur, _ = self.run(inputs=cur_docs_to_join, top_k_join=top_k_join)
                output_docs.append(cur["documents"])

            output = {"documents": output_docs, "labels": inputs[0].get("labels", None)}

            return output, "output_1"

    def _concatenate_results(self, results):
        """
        Concatenates multiple document result lists.
        """
        return {doc.id: doc.score for result in results for doc in result}

    def _calculate_comb_sum(self, results):
        """
        Calculates a combination sum by multiplying each score by its weight.
        """
        scores_map = defaultdict(int)
        weights = self.weights if self.weights else [1 / len(results)] * len(results)

        for result, weight in zip(results, weights):
            for doc in result:
                scores_map[doc.id] += (doc.score if doc.score else 0) * weight

        return scores_map

    def _calculate_rrf(self, results):
        """
        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        K = 61

        scores_map = defaultdict(int)
        for result in results:
            for rank, doc in enumerate(result):
                scores_map[doc.id] += 1 / (K + rank)

        return scores_map
