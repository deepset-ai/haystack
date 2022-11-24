from typing import Optional, List, Dict, Any, Union

import logging
from copy import deepcopy
from math import inf

from tqdm import tqdm

from haystack.schema import Document
from haystack.nodes.base import BaseComponent

logger = logging.getLogger(__name__)


class DocumentMerger(BaseComponent):
    """
    Merges all the documents into a single document.

    Retains all metadata that is present in all documents with the same value
    (for example, it retains the filename if all documents coming from the same file),

    Treats some metadata fields differently:
    - `headlines`: if `realign_headlines=True` (the default value), updates the content of the `start_idx` field of
        every headline to reflect the actual position in the merged document.
    - `page`: if `retain_page_number=True` (the default value), sets the value of the 'page' metadata field
        to the smallest value found across the documents to merge.
    """

    outgoing_edges = 1

    def __init__(
        self,
        separator: str = " ",
        window_size: int = 0,
        window_overlap: int = 0,
        max_tokens: Optional[int] = None,
        realign_headlines: bool = True,
        retain_page_number: bool = True,
        progress_bar: bool = True,
    ):
        """
        Merges the documents into one or more documents.

        Retains all metadata that is present in all documents with the same value
        (for example, it retains the filename if all documents coming from the same file),

        Treats some metadata fields differently:
        - `headlines`: if `realign_headlines=True` (the default value), updates the content of the `start_idx` field of
            every headline to reflect the actual position in the merged document.
        - `page`: if `retain_page_number=True` (the default value), sets the value of the 'page' metadata field
            to the smallest value found across the documents to merge.

        :param separator: A string that will be added between the contents of each merged document.
                          Might be a whitespace, a formfeed, a new line, an empty string, or any other string.

        :param window_size: The number of documents to include in each merged batch. For example, if set to 2,
                            the documents are merged in pairs. When set to 0, merges all documents into one
                            single document. Setting it to 0 can be useful along with max_tokens to try filling
                            documents by the largest amount of units that stays below the max_tokens value.

        :param window_overlap: Applies a sliding window approach over the documents groups. For example,
                               if `window_size=3` and `window_overlap=2`, the resulting documents come
                               from the merge of the following groups: `[doc1, doc2, doc3]`, `[doc2, doc3, doc4]`, ...

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it requires `tokenizer_model` to be set to the
                            model of your Reader and will verify that, whatever your `split_length` value is set
                            to, the number of tokens included in the split documents will never be above the
                            `max_tokens` value. For example:

                            ```python
                            DocumentMerger(window_size=10, max_tokens=512)
                            ```

                            means:

                            - Each merged document contains at most 10 source documents
                            - Merged documents might contain less than 10 source documents if the maximum number of tokens is
                                reached earlier.

                            The number of tokens might still be above the maximum if a single source document
                            contains more than 512 tokens. In this case an `ERROR` log is emitted, but the merged document
                            is generated with whatever amount of tokens the first source document has.

                            If the number of source documents is irrelevant, `window_size` can be safely set at `0`.

                            To use this parameter, each source document must have a metadata field such as
                            `"tokens_count": <int>`. This is usually created by the `DocumentSplitter`.

                            The `DocumentMerger`'s separator string will not be counted as a token. If you want to count it,
                            add the number of tokens it contains to each source document's `tokens_count` metadata field
                            to compensate.
        """
        super().__init__()
        self._validate_window_params(window_size=window_size, window_overlap=window_overlap)

        self.separator = separator
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.max_tokens = max_tokens
        self.realign_headlines = realign_headlines
        self.retain_page_number = retain_page_number
        self.progress_bar = progress_bar

    def _validate_window_params(self, window_size: int, window_overlap: int):
        """
        Performs basic validation on the values of window_size and window_overlap.
        """
        if window_size < 0 or not isinstance(window_size, int):
            raise ValueError("window_size must be an integer >= 0")

        if window_size:
            if window_overlap < 0 or not isinstance(window_overlap, int):
                raise ValueError("window_overlap must be an integer >= 0")

            if window_overlap >= window_size:
                raise ValueError("window_size must be larger than window_overlap")

    def run(  # type: ignore
        self,
        documents: List[Document],
        separator: Optional[str] = None,
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
        realign_headlines: Optional[bool] = None,
        retain_page_number: Optional[bool] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Merges the documents into one or more documents.

        Retains all metadata that is present in all documents with the same value
        (for example, it retains the filename if all documents coming from the same file),

        Treats some metadata fields differently:
        - `headlines`: if `realign_headlines=True` (the default value), updates the content of the `start_idx` field of
            every headline to reflect the actual position in the merged document.
        - `page`: if `retain_page_number=True` (the default value), sets the value of the 'page' metadata field
            to the smallest value found across the documents to merge.

        :param separator: A string that will be added between the contents of each merged document.
                          Might be a whitespace, a formfeed, a new line, an empty string, or any other string.

        :param window_size: The number of documents to include in each merged batch. For example, if set to 2,
                            the documents are merged in pairs. When set to 0, merges all documents into one
                            single document.

        :param window_overlap: Applies a sliding window approach over the documents groups. For example,
                               if `window_size=3` and `window_overlap=2`, the resulting documents come
                               from the merge of the following groups: `[doc1, doc2, doc3]`, `[doc2, doc3, doc4]`, ...

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it requires `tokenizer_model` to be set to the
                            model of your Reader and will verify that, whatever your `split_length` value is set
                            to, the number of tokens included in the split documents will never be above the
                            `max_tokens` value. For example:

                            ```python
                            DocumentMerger(window_size=10, max_tokens=512)
                            ```

                            means:

                            - Each merged document contains at most 10 source documents
                            - Merged documents might contain less than 10 source documents if the maximum number of tokens is
                                reached earlier.

                            The number of tokens might still be above the maximum if a single source document
                            contains more than 512 tokens. In this case an `ERROR` log is emitted, but the merged document
                            is generated with whatever amount of tokens the first source document has.

                            If the number of source documents is irrelevant, `window_size` can be safely set at `0`.

                            To use this parameter, each source document must have a metadata field such as
                            `"tokens_count": <int>`. This is usually created by the `DocumentSplitter`.

                            The `DocumentMerger`'s separator string will not be counted as a token. If you want to count it,
                            add the number of tokens it contains to each source document's `tokens_count` metadata field
                            to compensate.
        """
        if not all(doc.content_type == "text" for doc in documents):
            raise ValueError(
                "DocumentMerger received some documents that do not contain text. "
                "Make sure to pass only text documents to it. "
                "You can use a RouteDocuments node to make sure only text document are sent to the DocumentMerger."
            )

        # For safety, as we manipulate the meta
        documents = deepcopy(documents)

        separator = separator if separator is not None else self.separator
        window_size = window_size if window_size is not None else self.window_size
        window_overlap = window_overlap if window_overlap is not None else self.window_overlap
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        realign_headlines = realign_headlines if realign_headlines is not None else self.realign_headlines
        retain_page_number = retain_page_number if retain_page_number is not None else self.retain_page_number
        self._validate_window_params(window_size=window_size, window_overlap=window_overlap)

        # Create the groups according to max_tokens, window_size and window_overlap
        if max_tokens:
            merged_documents = []
            group: List[Document] = []
            tokens_count = 0
            documents_added = 0
            doc_index = 0
            while doc_index < len(documents):

                if tokens_count + documents[doc_index].meta["tokens_count"] > max_tokens or documents_added >= (
                    window_size or inf
                ):
                    if len(group) == 1 and tokens_count > max_tokens:
                        logger.error(
                            "Seems like one document has more tokens than the allowed `max_tokens` value: %s > %s. "
                            "This document will be created with %s tokens. Consider reducing window_size "
                            "(split_length in DocumentSplitter)",
                            tokens_count,
                            max_tokens,
                            tokens_count,
                        )

                    merged_documents.append(
                        self.merge(
                            group=group,
                            separator=separator,
                            realign_headlines=realign_headlines,
                            retain_page_number=retain_page_number,
                            tokens_count=tokens_count,
                        )
                    )
                    group = []
                    tokens_count = 0
                    documents_added = 0
                    if window_overlap:
                        doc_index -= window_overlap

                tokens_count += documents[doc_index].meta["tokens_count"]
                documents_added += 1
                group.append(documents[doc_index])
                doc_index += 1

            if group:
                merged_documents.append(
                    self.merge(
                        group=group,
                        separator=separator,
                        realign_headlines=realign_headlines,
                        retain_page_number=retain_page_number,
                        tokens_count=tokens_count,
                    )
                )

        # Shortcuts for when max_tokens is not used
        else:
            if window_size:
                groups = [
                    documents[pos : pos + window_size]
                    for pos in range(0, max(1, len(documents) - window_overlap), window_size - window_overlap)
                ]
            else:
                groups = [documents]

            merged_documents = [
                self.merge(
                    group=group,
                    separator=separator,
                    realign_headlines=realign_headlines,
                    retain_page_number=retain_page_number,
                )
                for group in groups
                if group
            ]
        return {"documents": merged_documents}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        separator: Optional[str] = None,
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
        realign_headlines: Optional[bool] = None,
        retain_page_number: Optional[bool] = None,
    ):
        """
        Merges the documents into one or more documents.

        Retains all metadata that is present in all documents with the same value
        (for example, it retains the filename if all documents coming from the same file),

        Treats some metadata fields differently:
        - `headlines`: if `realign_headlines=True` (the default value), updates the content of the `start_idx` field of
            every headline to reflect the actual position in the merged document.
        - `page`: if `retain_page_number=True` (the default value), sets the value of the 'page' metadata field
            to the smallest value found across the documents to merge.

        :param separator: A string that will be added between the contents of each merged document.
                          Might be a whitespace, a formfeed, a new line, an empty string, or any other string.
        :param window_size: The number of documents to include in each merged batch. For example, if set to 2,
                            the documents are merged in pairs. When set to 0, merges all documents into one
                            single document.
        :param window_overlap: Applies a sliding window approach over the documents groups. For example,
                               if `window_size=3` and `window_overlap=2`, the resulting documents come
                               from the merge of the following groups: `[doc1, doc2, doc3]`, `[doc2, doc3, doc4]`, ...
        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.
        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.
        """
        result = [
            self.run(
                documents=docs,
                separator=separator,
                window_size=window_size,
                window_overlap=window_overlap,
                realign_headlines=realign_headlines,
                retain_page_number=retain_page_number,
            )[0]["documents"]
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Merging", unit="docs")
        ]
        return {"documents": result}, "output_1"

    def merge(
        self,
        group: List[Document],
        separator: str,
        realign_headlines: bool = True,
        retain_page_number: bool = True,
        tokens_count: Optional[int] = None,
    ) -> Document:
        """
        Merges the documents into one documents.

        Retains all metadata that is present in all documents with the same value
        (for example, it retains the filename if all documents coming from the same file),

        Treats some metadata fields differently:
        - `headlines`: if `realign_headlines=True` (the default value), updates the content of the `start_idx` field of
            every headline to reflect the actual position in the merged document.
        - `page`: if `retain_page_number=True` (the default value), sets the value of the 'page' metadata field
            to the smallest value found across the documents to merge.

        :param group: The list of documents to merge together
        :param separator: A string that will be added between the contents of each merged document.
                          Might be a whitespace, a formfeed, a new line, an empty string, or any other string.
        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.
        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.
        :param tokens_count: The cumulative number of tokens contained in this group, if relevant.
        """
        if not group:
            raise ValueError(
                "No documents in the `group` parameter. "
                "Make sure to pass some documents to the DocumentMerger.merge() method."
            )

        merged_content = separator.join([doc.content for doc in group])
        merged_document = Document(content=merged_content)

        # Realign headlines or erase them
        headlines_meta = {}
        if realign_headlines and any("headlines" in doc.meta.keys() for doc in group):
            if any(doc.meta["headlines"] is not None for doc in group):
                merged_headlines = merge_headlines(documents=group, separator=separator)
                headlines_meta = {"headlines": merged_headlines}
        else:
            for doc in group:
                if "headlines" in doc.meta.keys():
                    del doc.meta["headlines"]

        # Reset page number or erase it
        page_number_meta = {}
        if retain_page_number and any("page" in doc.meta.keys() for doc in group):
            page_number_meta = {"page": min(int(doc.meta.get("page", inf)) for doc in group)}
        else:
            for doc in group:
                if "page" in doc.meta.keys():
                    del doc.meta["page"]

        # Retain any other common key
        merged_document.meta = {**common_values([doc.meta for doc in group]), **headlines_meta, **page_number_meta}

        if tokens_count:
            merged_document.meta["tokens_count"] = tokens_count

        return merged_document


def merge_headlines(documents: List[Document], separator: str) -> List[Dict[str, Union[str, int]]]:
    """
    Merges the headlines dictionary with the new position of each headline into the merged document.
    Assumes the documents are in the same order as when they were merged.
    """
    aligned_headlines = []
    position_in_merged_document = 0
    for doc in documents:
        if doc.meta.get("headlines", []):
            for headline in deepcopy(doc.meta.get("headlines", [])):
                headline["start_idx"] += position_in_merged_document
                aligned_headlines.append(headline)
        position_in_merged_document += len(doc.content) + len(separator)
    return aligned_headlines


def common_values(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Retains all keys shared across all the documents being merged.

    Such keys are checked recursively, see tests.
    """
    merge_dictionary = deepcopy(list_of_dicts[0])
    for key, value in list_of_dicts[0].items():

        # if not all other dicts have this key, delete directly
        if not all(key in dict.keys() for dict in list_of_dicts):
            del merge_dictionary[key]

        # if they all have it and it's a dictionary, merge recursively
        elif isinstance(value, dict):
            # Get all the subkeys to merge in a new list
            list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
            merge_dictionary[key] = common_values(list_of_subdicts)

        # If all dicts have this key and it's not a dictionary, delete only if the values differ
        elif not all(value == dict[key] for dict in list_of_dicts):
            del merge_dictionary[key]

    return merge_dictionary or {}
