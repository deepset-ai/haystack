from collections import defaultdict
import logging
from math import inf

from typing import Optional, List

from haystack.schema import Document
from haystack.nodes import BaseNode

logger = logging.getLogger(__name__)


class DocumentMerger(BaseNode):
    """
    A node to merge the texts of the documents.
    """

    outgoing_edges = 1

    def __init__(self, separator: str = " "):
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
        super().__init__()

        self.separator = separator

    def merge(self, documents: List[Document], separator: Optional[str]) -> List[Document]:

        if len(documents) == 0:
            raise AttributeError("Document Merger needs at least one document to merge.")

        if not all([doc.content_type == "text" for doc in documents]):
            raise AttributeError(
                "Not all the provided documents are textual. Document Merger only works on textual documents."
            )

        if separator is None:
            separator = self.separator

        merged_content = separator.join([doc.content for doc in documents])

        # better with other solutions?
        # how to approach nested?
        common_meta_dict = deepcopy(documents[0].meta)
        for doc in documents[1:]:
            if len(common_meta_dict) == 0:
                break
            for k, v in doc.meta.items():
                if k in common_meta_dict:
                    if common_meta_dict[k] != v:
                        del common_meta_dict[k]

        # common_meta_keys = set(documents[0].meta.keys())
        # for doc in documents[1:]:
        #     common_meta_keys.intersection_update(set(doc.meta.keys()))

        # for doc in documents:
        #     common_meta_keys=set.intersection(*map(set, my_dict))
        #     for k,v doc.meta.items()

        return None

    def run(self, documents: List[Document], separator: Optional[str]):  # type: ignore
        """Method that gets executed when this class is used as a Node in a Haystack Pipeline"""
        return None

    def run_batch(  # type: ignore
        self,
        documents: Union[List[Document], List[List[Document]]],
        separator: Optional[str],
        batch_size: Optional[int] = None,
    ):

        return None, "output_1"
