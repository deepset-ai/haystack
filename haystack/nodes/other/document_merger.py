from collections import MutableMapping
import logging
from copy import deepcopy
from typing import Optional, List, Dict, Union

from haystack.schema import Document
from haystack.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class DocumentMerger(BaseComponent):
    """
    A node to merge the texts of the documents.
    """

    outgoing_edges = 1

    def __init__(self, separator: str = " "):
        """
        :param separator: The separator that appears between subsequent merged documents.
        """
        super().__init__()
        self.separator = separator

    def merge(self, documents: List[Document], separator: Optional[str] = None) -> List[Document]:
        """
        Produce a list made up of a single document, which contains all the texts of the documents provided.

        :param separator: The separator that appears between subsequent merged documents.
        :return: List of Documents
        """
        if len(documents) == 0:
            raise AttributeError("Document Merger needs at least one document to merge.")
        if not all([doc.content_type == "text" for doc in documents]):
            raise AttributeError(
                "Some of the documents provided are non-textual. Document Merger only works on textual documents."
            )

        if separator is None:
            separator = self.separator

        merged_content = separator.join([doc.content for doc in documents])
        common_meta = self._extract_common_meta_dict(documents)

        merged_document = Document(content=merged_content, meta=common_meta)
        return [merged_document]

    def run(self, documents: List[Document], separator: Optional[str] = None):  # type: ignore
        results: Dict = {"documents": []}
        if documents:
            results["documents"] = self.merge(documents=documents, separator=separator)
        return results, "output_1"

    def run_batch(  # type: ignore
        self, documents: Union[List[Document], List[List[Document]]], separator: Optional[str] = None
    ):
        is_doclist_flat = isinstance(documents[0], Document)
        if is_doclist_flat:
            flat_result: List[Document] = []
            flat_result = self.merge(documents=documents, separator=separator)
            return {"documents": flat_result}, "output_1"
        else:
            nested_result: List[List[Document]] = []
            for docs_group in documents:
                nested_result.append(self.merge(documents=docs_group, separator=separator))
            return {"documents": nested_result}, "output_1"

    def _extract_common_meta_dict(self, documents: List[Document]) -> dict:
        """
        Given a list of documents, extract a dictionary containing the meta fields
        that are common to all the documents
        """
        flattened_meta = [self._flatten_dict(d.meta) for d in documents]
        common_meta_flat_dict = deepcopy(flattened_meta[0])
        for doc in flattened_meta[1:]:
            if len(common_meta_flat_dict) == 0:
                break
            for k, v in doc.items():
                if k in common_meta_flat_dict:
                    if common_meta_flat_dict[k] != v:
                        del common_meta_flat_dict[k]
        common_meta_nested_dict = self._nest_dict(common_meta_flat_dict)
        return common_meta_nested_dict

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
        nested_dict: dict = {}
        for key, value in d.items():
            target = nested_dict
            if isinstance(key, tuple):
                for k in key[:-1]:  # traverse all keys but the last
                    target = target.setdefault(k, {})
                target[key[-1]] = value
            else:
                target[key] = value
        while any(isinstance(k, tuple) for k in nested_dict.keys()):
            nested_dict = self._nest_dict(nested_dict)
        return nested_dict
