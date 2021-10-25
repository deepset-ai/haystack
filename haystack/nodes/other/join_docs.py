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
    """
    outgoing_edges = 1

    def __init__(
        self, join_mode: str = "concatenate", weights: Optional[List[float]] = None, top_k_join: Optional[int] = None
    ):
        """
        :param join_mode: `concatenate` to combine documents from multiple retrievers or `merge` to aggregate scores of
                          individual documents.
        :param weights: A node-wise list(length of list must be equal to the number of input nodes) of weights for
                        adjusting document scores when using the `merge` join_mode. By default, equal weight is given
                        to each retriever score. This param is not compatible with the `concatenate` join_mode.
        :param top_k_join: Limit documents to top_k based on the resulting scores of the join.
        """
        assert join_mode in ["concatenate", "merge"], f"JoinDocuments node does not support '{join_mode}' join_mode."

        assert not (
            weights is not None and join_mode == "concatenate"
        ), "Weights are not compatible with 'concatenate' join_mode."

        # save init parameters to enable export of component config as YAML
        self.set_config(join_mode=join_mode, weights=weights, top_k_join=top_k_join)

        self.join_mode = join_mode
        self.weights = [float(i)/sum(weights) for i in weights] if weights else None
        self.top_k_join = top_k_join

    def run(self, inputs: List[dict]):  # type: ignore
        if self.join_mode == "concatenate":
            document_map = {}
            for input_from_node in inputs:
                for doc in input_from_node["documents"]:
                    document_map[doc.id] = doc
        elif self.join_mode == "merge":
            document_map = {}
            if self.weights:
                weights = self.weights
            else:
                weights = [1/len(inputs)] * len(inputs)
            for input_from_node, weight in zip(inputs, weights):
                for doc in input_from_node["documents"]:
                    if document_map.get(doc.id):  # document already exists; update score
                        document_map[doc.id].score += doc.score * weight
                    else:  # add the document in map
                        document_map[doc.id] = deepcopy(doc)
                        document_map[doc.id].score *= weight
        else:
            raise Exception(f"Invalid join_mode: {self.join_mode}")

        documents = sorted(document_map.values(), key=lambda d: d.score, reverse=True)

        if self.top_k_join:
            documents = documents[: self.top_k_join]
        output = {"documents": documents, "labels": inputs[0].get("labels", None)}
        return output, "output_1"
