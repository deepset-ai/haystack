from collections import MutableMapping
import logging
from math import inf
from copy import deepcopy

from typing import Optional, List, Dict, Any

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
                "Some of the documents provided are non-textual. Document Merger only works on textual documents."
            )

        if separator is None:
            separator = self.separator

        merged_content = separator.join([doc.content for doc in documents])

        flattened_meta = [self._flatten_dict(d["meta"]) for d in documents]

        common_meta_flat_dict = deepcopy(flattened_meta[0])
        for doc in flattened_meta[1:]:
            if len(common_meta_flat_dict) == 0:
                break
            for k, v in doc.items():
                if k in common_meta_flat_dict:
                    if common_meta_flat_dict[k] != v:
                        del common_meta_flat_dict[k]
        common_meta_nest_dict = self._nest_dict(common_meta_flat_dict)

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

    def _flatten_dict(self, d: dict, parent_key="") -> dict:
        items: List = []
        for k, v in d.items():
            new_key = (parent_key, k) if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _nest_dict(self, d: dict) -> dict:
        nested_dict = {}
        for key, value in d.items():
            target = nested_dict
            if isinstance(key, tuple):
                for k in key[:-1]:  # traverse all keys but the last
                    target = target.setdefault(k, {})
                target[key[-1]] = value
            else:
                target[key] = value
        while any(isinstance(k, tuple) for k in nested_dict.keys()):
            nested_dict = nest(nested_dict)
        return common_meta_nested_dict
