import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional

from haystack.preview import component, default_from_dict, default_to_dict, Document


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

        :param join_mode:
        :param weights:
        :param top_k: How many documents should be returned in the output at maximum. By default, all are returned.
        :param sort_by_score:
        """
        if join_mode not in ["concatenate", "merge", "reciprocal_rank_fusion"]:
            raise ValueError(f"DocumentJoiner component does not support '{join_mode}' join_mode.")
        self.join_mode = join_mode
        self.weights = [float(i) / sum(weights) for i in weights] if weights else None
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(documents=List[Document])
    def run(self, *documents: List[Document]):
        """
        Run the DocumentJoiner. This method joins the input lists of documents to into one output list based on the join_mode specified during initialization.

        :param documents: A list of documents to route to different edges.
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
                output_documents, key=lambda doc: doc.score if doc.score is not None else -1, reverse=True
            )
        if self.top_k:
            output_documents = output_documents[: self.top_k]
        return {"documents": output_documents}

    def _concatenate(self, *documents):
        """
        Concatenates multiple lists of documents into one list of documents.
        """
        return list(itertools.chain.from_iterable(*documents))

    def _merge(self, *documents):
        """
        Concatenates multiple lists of documents into one list of documents.
        """
        return list(itertools.chain.from_iterable(*documents))

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self, join_mode=self.join_mode, weights=self.weights, top_k=self.top_k, sort_by_score=self.sort_by_score
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentJoiner":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)
