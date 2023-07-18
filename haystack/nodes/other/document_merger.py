import logging
from copy import deepcopy
from typing import Optional, List, Dict, Union, Any

from haystack.schema import Document
from haystack.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


def truncate_document(
    document: str, separator: Optional[str] = None, word_count_threshold: Optional[int] = None
) -> str:
    """
    Shortens a document by cutting off the content after a specified number of words.

    :param document: String representing the content of the document.
    :param separator: Optional string representing the separator between words in the document.
    :param word_count_threshold: Optional integer representing the maximum number of words
                                 allowed in the truncated document. If None or less than 1,
                                 no truncation will occur.

    :return: A string containing the truncated document. If the word count threshold is not
             specified or is less than 1, the original document is returned.
    """
    separator = separator or " "
    return (
        separator.join(document.split()[:word_count_threshold])
        if word_count_threshold and word_count_threshold > 0
        else document
    )


def default_order(
    documents: List[Document], separator: Optional[str] = None, word_count_threshold: Optional[int] = None
) -> str:
    """
    Merges documents in the order they are provided.

    :param documents: List of Documents to merge.
    :param separator: The separator that appears between subsequent merged documents.
    :param word_count_threshold: The maximum number of words in the merged document.

    :return: The merged document as a string.
    """
    separator = separator or " "
    merged_doc_content = separator.join([doc.content for doc in documents])
    return truncate_document(merged_doc_content, separator, word_count_threshold)


def lost_in_the_middle_order(
    documents: List[Document], separator: Optional[str] = None, word_count_threshold: Optional[int] = None
) -> str:
    """
    Merges documents as suggested by "Lost in the Middle: How Language Models Use Long Contexts" by Liu et al.
    Liu et al. suggest to layout paragraphs into LLM context so that relevant paragraphs at the beginning
    or end of the input context, while the least relevant information should be in the middle of a context.

    See https://arxiv.org/abs/2307.03172 for more details.

    :param documents: List of Documents to merge.
    :param separator: The separator that appears between subsequent merged documents.
    :param word_count_threshold: The maximum number of words in the merged document.

    :return: The merged document as a string.
    """
    if not documents:
        return ""
    if len(documents) == 1:
        return truncate_document(documents[0].content, separator, word_count_threshold)

    document_index = [i for i in range(len(documents))]

    merged_doc_content = ""
    separator = separator or " "
    lost_in_the_middle_indices = [0]
    for doc in document_index[1:]:
        insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2
        lost_in_the_middle_indices.insert(insertion_index, doc)

        merged_doc_content = separator.join([documents[index].content for index in lost_in_the_middle_indices])

        # While it might seem intuitive to first order the documents and subsequently truncate the combined
        # document, this approach would yield an inaccurate order. Instead, we need to truncate the documents
        # concurrently with their arrangement in the "lost in the middle" order, as this dynamic layout process
        # determines the correct sequencing of the documents while respecting the word count threshold.
        if word_count_threshold and len(merged_doc_content.split()) >= word_count_threshold:
            merged_doc_content = truncate_document(merged_doc_content, separator, word_count_threshold)
            break

    return merged_doc_content.strip()


class DocumentMerger(BaseComponent):
    """
    A node to merge the texts of the documents.
    """

    outgoing_edges = 1

    def __init__(self, separator: str = " ", order: str = "default", word_count_threshold: Optional[int] = None):
        """
        :param separator: The separator that appears between subsequent merged documents.
        :param order: The order in which the documents are merged.
        :param word_count_threshold: Optional integer representing the maximum number of words in the merged document.
        """
        super().__init__()
        self.separator = separator
        self.word_count_threshold = word_count_threshold
        self.layout_algorithms = {"default": default_order, "lost_in_the_middle": lost_in_the_middle_order}
        if order not in self.layout_algorithms:
            raise ValueError(
                f"Unknown DocumentMerger order {order}. Possible values are {list(self.layout_algorithms.keys())}."
            )
        self.order = order

    def merge(self, documents: List[Document], separator: Optional[str] = None) -> List[Document]:
        """
        Produce a list made up of a single document, which contains all the texts of the documents provided.

        :param documents: List of Documents to merge.
        :param separator: The separator that appears between subsequent merged documents.
        :return: List of Documents
        """
        if len(documents) == 0:
            raise ValueError("Document Merger needs at least one document to merge.")
        if not all(doc.content_type == "text" for doc in documents):
            raise ValueError(
                "Some of the documents provided are non-textual. Document Merger only works on textual documents."
            )

        separator = separator if separator is not None else self.separator

        merged_content = self.layout_algorithms[self.order](
            documents=documents, separator=separator, word_count_threshold=self.word_count_threshold
        )
        common_meta = self._keep_common_keys([doc.meta for doc in documents])

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
            flat_result: List[Document] = self.merge(
                documents=[doc for doc in documents if isinstance(doc, Document)], separator=separator
            )
            return {"documents": flat_result}, "output_1"
        else:
            nested_result: List[List[Document]] = [
                self.merge(documents=docs_lst, separator=separator)
                for docs_lst in documents
                if isinstance(docs_lst, list)
            ]
            return {"documents": nested_result}, "output_1"

    def _keep_common_keys(self, list_of_dicts: List[Dict[str, Any]]) -> dict:
        merge_dictionary = deepcopy(list_of_dicts[0])
        for key, value in list_of_dicts[0].items():
            # if not all other dicts have this key, delete directly
            if not all(key in dict.keys() for dict in list_of_dicts):
                del merge_dictionary[key]

            # if they all have it and it's a dictionary, merge recursively
            elif isinstance(value, dict):
                # Get all the subkeys to merge in a new list
                list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
                merge_dictionary[key] = self._keep_common_keys(list_of_subdicts)

            # If all dicts have this key and it's not a dictionary, delete only if the values differ
            elif not all(value == dict[key] for dict in list_of_dicts):
                del merge_dictionary[key]

        return merge_dictionary
