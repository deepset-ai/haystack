import itertools

from copy import deepcopy
from typing import Optional, List

from haystack.nodes.base import BaseComponent


class JoinDocuments(BaseComponent):
    """
    A node to join documents outputted by multiple retriever nodes.

    The node allows multiple join modes:
    * concatenate: combine the documents from multiple nodes. Any duplicate documents are discarded.
    * merge: merge scores of documents from multiple nodes. Optionally, each input score can be given a different
             `weight` & a `top_k` limit can be set. This mode can also be used for "reranking" retrieved documents.
    * reciprocal_rank_fusion: combines the documents based on their rank in multiple nodes.
    """

    outgoing_edges = 1

    def __init__(
        self, join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None
    ):
        """
        :param join_mode: `concatenate` to combine documents from multiple retrievers `merge` to aggregate scores of
                          individual documents, `reciprocal_rank_fusion` to apply rank based scoring.
        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                        to each retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k_join: Limit documents to top_k based on the resulting scores of the join.
        """
        assert join_mode in ["concatenate", "merge", "reciprocal_rank_fusion"], f"JoinDocuments node does not support '{join_mode}' join_mode."

        assert not (
            weights is not None and join_mode == "concatenate"
        ), "Weights are not compatible with 'concatenate' join_mode."

        # save init parameters to enable export of component config as YAML
        self.set_config(join_mode=join_mode, weights=weights, top_k_join=top_k_join)

        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join

    def run(self, inputs: List[dict], top_k_join: Optional[int] = None):  # type: ignore
        results = [inp["documents"] for inp in inputs]
        DocScore = namedtuple("DocScore", field_names="document,score")
        weights = self.weights if self.weights else [1 / len(inputs)] * len(inputs)
        document_map = {doc.id: DocScore(document=doc, score=0) for doc in itertools.chain(*results)}

        for result, weight in zip(results, weights):
            for rank, doc in enumerate(result):
                if self.join_mode == "concatenate":
                    document_map[doc.id].score = doc.score
                elif self.join_mode == "merge":
                    document_map[doc.id].score += self._calculate_comb_sum(doc, weight)
                elif self.join_mode == "reciprocal_rank_fusion":
                    document_map[doc.id].score += self._calculate_rrf(rank)
                else:
                    raise Exception(f"Invalid join_mode: {self.join_mode}")

        sorted_docs = sorted(document_map.values(), key=lambda d: d.score, reverse=True)
        docs = []
        for document_tuple in sorted_docs:
            doc = document_tuple.document
            doc.score = document_tuple.score
            docs.append(doc)

        top_k_join = top_k_join if top_k_join else self.top_k_join
        if top_k_join:
            docs = docs[:top_k_join]

        output = {"documents": docs, "labels": inputs[0].get("labels", None)}

        return output, "output_1"

    def _calculate_comb_sum(self, document, weight):
        return document.score * weight

    def _calculate_rrf(self, rank):
        return 1 / (61 + rank)
