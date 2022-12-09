from typing import Optional, List, Dict, Any, Union, Tuple

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
        max_chars: int = 0,
        max_tokens: int = 0,
        merge_metadata: bool = True,
        realign_headlines: bool = True,
        retain_page_number: bool = True,
        progress_bar: bool = True,
        id_hash_keys: Optional[List[str]] = None,
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

        :param merge_metadata: Whether the merged document should try to merge the metadata of the incoming documents.
                                Merging means retaining all and only the keys whose values are identical across all the
                                input documents (for example filename, language, etc).

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error. This parameter has higher priority than
                            both `window_size` and `max_tokens`.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.

        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it checks that, whatever your `split_length` value is set
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

        :param id_hash_keys: the value to pass to id_hash_keys when initializing the merged Document(s).
        """
        super().__init__()
        self._validate_window_params(window_size=window_size, window_overlap=window_overlap)

        self.separator = separator
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.merge_metadata = merge_metadata
        self.realign_headlines = realign_headlines
        self.retain_page_number = retain_page_number
        self.progress_bar = progress_bar
        self.id_hash_keys = id_hash_keys

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
        merge_metadata: Optional[bool] = None,
        realign_headlines: Optional[bool] = None,
        retain_page_number: Optional[bool] = None,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        id_hash_keys: Optional[List[str]] = None,
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

        :param merge_metadata: Whether the merged document should try to merge the metadata of the incoming documents.
                                Merging means retaining all and only the keys whose values are identical across all the
                                input documents (for example filename, language, etc).

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error. This parameter has higher priority than
                            both `window_size` and `max_tokens`.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.

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

        :param id_hash_keys: the value to pass to id_hash_keys when initializing the merged Document(s).
        """
        merged_documents = self.merge(
            documents=documents,
            separator=separator,
            window_size=window_size,
            window_overlap=window_overlap,
            merge_metadata=merge_metadata,
            realign_headlines=realign_headlines,
            retain_page_number=retain_page_number,
            max_tokens=max_tokens,
            max_chars=max_chars,
            id_hash_keys=id_hash_keys,
        )
        return {"documents": merged_documents}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        separator: Optional[str] = None,
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_chars: Optional[int] = None,
        merge_metadata: Optional[bool] = None,
        realign_headlines: Optional[bool] = None,
        retain_page_number: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
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

        :param merge_metadata: Whether the merged document should try to merge the metadata of the incoming documents.
                                Merging means retaining all and only the keys whose values are identical across all the
                                input documents (for example filename, language, etc).

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error. This parameter has higher priority than
                            both `window_size` and `max_tokens`.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.

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

        :param id_hash_keys: the value to pass to id_hash_keys when initializing the merged Document(s).
        """
        result = [
            self.run(
                documents=docs,
                separator=separator,
                window_size=window_size,
                window_overlap=window_overlap,
                max_tokens=max_tokens,
                max_chars=max_chars,
                merge_metadata=merge_metadata,
                realign_headlines=realign_headlines,
                retain_page_number=retain_page_number,
                id_hash_keys=id_hash_keys,
            )[0]["documents"]
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Merging", unit="docs")
        ]
        return {"documents": result}, "output_1"

    def merge(
        self,
        documents: List[Document],
        separator: Optional[str] = None,
        window_size: Optional[int] = None,
        window_overlap: Optional[int] = None,
        merge_metadata: Optional[bool] = None,
        realign_headlines: Optional[bool] = None,
        retain_page_number: Optional[bool] = None,
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
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

        :param merge_metadata: Whether the merged document should try to merge the metadata of the incoming documents.
                                Merging means retaining all and only the keys whose values are identical across all the
                                input documents (for example filename, language, etc).

        :param realign_headlines: Whether to update the value of `start_idx` for the document's headlines, if found
                                  in the metadata. Setting it to `False` drops all the headline information found.

        :param retain_page_number: Whether to set the page number to the lowest value in case of mismatch across the
                                   merged documents. Setting it to `False` always drops the page number from the
                                   merged document.

        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error. This parameter has higher priority than
                            both `window_size` and `max_tokens`.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.

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

        :param id_hash_keys: the value to pass to id_hash_keys when initializing the merged Document(s).
        """
        if not documents:
            return []

        if not all(doc.content_type == "text" for doc in documents):
            raise ValueError(
                "DocumentMerger received some documents that do not contain text. "
                "Make sure to pass only text documents to it. "
                "You can use a RouteDocuments node to make sure only text documents are sent to the DocumentMerger."
            )

        separator = separator if separator is not None else self.separator
        window_size = window_size if window_size is not None else self.window_size
        window_overlap = window_overlap if window_overlap is not None else self.window_overlap
        max_chars = max_chars if max_chars is not None else self.max_chars
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        merge_metadata = merge_metadata if merge_metadata is not None else self.merge_metadata
        realign_headlines = realign_headlines if realign_headlines is not None else self.realign_headlines
        retain_page_number = retain_page_number if retain_page_number is not None else self.retain_page_number
        id_hash_keys = id_hash_keys if id_hash_keys is not None else self.id_hash_keys
        self._validate_window_params(window_size=window_size, window_overlap=window_overlap)

        valid_contents = validate_boundaries(
            contents=[doc.content for doc in documents], max_chars=max_chars, max_tokens=max_tokens, tokens=tokens
        )
        groups_to_merge = self.make_groups(
            contents=valid_contents,
            window_size=window_size,
            window_overlap=window_overlap,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

        merged_documents = []
        for group in groups_to_merge:
            merged_content = separator.join([valid_contents[doc_index][0] for doc_index in group])
            merged_documents.append(Document(content=merged_content, id_hash_keys=id_hash_keys))

        if merge_metadata:
            for group_index, group in enumerate(groups_to_merge):
                metas = [documents[doc_index].meta for doc_index in group]
                merged_documents[group_index].meta = {**common_values(metas, exclude=["headlines", "page"])}

        if realign_headlines:
            for group_index, group in enumerate(groups_to_merge):
                sources = [
                    (documents[doc_index].content, deepcopy(documents[doc_index].meta.get("headlines")) or [])
                    for doc_index in group
                ]
                merged_headlines = merge_headlines(sources=sources, separator=separator)
                if merged_headlines:
                    merged_documents[group_index].meta["headlines"] = merged_headlines

        if retain_page_number:
            for group_index, group in enumerate(groups_to_merge):
                page = min(int(documents[doc_index].meta.get("page", inf)) for doc_index in group)
                if page != inf:
                    merged_documents[group_index].meta["page"] = page

        return merged_documents

    def make_groups(
        self, contents: List[Tuple[str, int]], window_size: int, window_overlap: int, max_tokens: int, max_chars: int
    ):
        """
        Creates the groups of documents that need to be merged, respecting all boundaries.

        :param contents: a tuple with the document content and how many tokens it contains.
            Can be created with `validate_boundaries()`
        :param window_size: The number of documents to include in each merged batch. For example,
            if set to 2, the documents are merged in pairs. When set to 0, merges all documents
            into one single document.
        :param window_overlap: Applies a sliding window approach over the documents groups.
            For example, if `window_size=3` and `window_overlap=2`, the resulting documents come
            from the merge of the following groups: `[doc1, doc2, doc3]`, `[doc2, doc3, doc4]`, ...
        :param max_tokens: the maximum number of tokens allowed
        :param max_chars: the maximum number of chars allowed
        """
        groups = []
        if max_tokens:
            group: List[int] = []
            content_len = 0
            tokens_count = 0
            doc_index = 0
            while doc_index < len(contents):

                if max_chars and content_len + len(contents[doc_index][0]) > max_chars:
                    # Rare and odd case: log loud
                    logging.warning(
                        "One document reached `max_chars` (%s) before either `max_tokens` (%s) or `window_size` (%s). "
                        "The last unit is moved to the next document to keep the chars count below the threshold."
                        "Consider raising `max_chars` and double-check how the input coming from earlier nodes looks like.",
                        max_chars,
                        max_tokens,
                        window_size,
                    )
                    groups.append(group)
                    group = []
                    tokens_count = 0
                    content_len = 0
                    if window_overlap:
                        doc_index -= window_overlap

                if max_tokens and tokens_count + contents[doc_index][1] > max_tokens:
                    # More plausible case: log, but not too loud
                    logging.info(
                        "One document reached `max_tokens` (%s) before `window_size` (%s). "
                        "The last unit is moved to the next document to keep the token count below the threshold.",
                        max_tokens,
                        window_size,
                    )
                    groups.append(group)
                    group = []
                    tokens_count = 0
                    content_len = 0
                    if window_overlap:
                        doc_index -= window_overlap

                if window_size and len(group) >= window_size:
                    # Fully normal: debug log only
                    logging.debug(
                        "One document reached `window_size` (%s) before `max_tokens` (%s). ", max_tokens, window_size
                    )
                    groups.append(group)
                    group = []
                    tokens_count = 0
                    content_len = 0
                    if window_overlap:
                        doc_index -= window_overlap

                # Still accumulating
                group.append(doc_index)
                tokens_count += contents[doc_index][1]
                content_len += len(contents[doc_index][0])
                doc_index += 1

            # Last group after the loop
            if group:
                group.append(doc_index)
            return group

        # Shortcuts for when max_tokens is not used
        elif window_size:
            return [
                list(range(pos, pos + window_size))
                for pos in range(0, max(1, len(contents) - window_overlap), window_size - window_overlap)
            ]
        else:
            return [list(range(len(contents)))]


def validate_boundaries(
    contents: List[str], max_chars: int, max_tokens: int, tokens: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    """
    Makes sure all boundaries (max_tokens if given, max_char if given) are respected. Splits the strings if necessary.

    :param contents: the content of all documents to merge, as a string
    :param tokens: a single list with all the tokens contained in all documents to merge.
        So, if `contents = ["hello how are you", "I'm fine thanks"]`,
        tokens should contain something similar to `["hello", "how", "are", "you", "I'm", "fine", "thanks"]`
    :param max_tokens: the maximum amount of tokens allowed in a doc
    :param max_chars: the maximum number of chars allowed in a doc
    :return: a tuple (content, n_of tokens) if max_token is set, else (content, 0)
    """
    valid_contents = []

    # Count tokens and chars, split if necessary
    if max_tokens:
        if not tokens:
            raise ValueError("if max_tokens is set, you must pass the tokenized text to `tokens`.")

        for content in contents:
            tokens_length = 0
            for tokens_count, token in enumerate(tokens):
                tokens_length += len(token)

                # If we reached the doc length, record how many tokens it contained and pass on the next doc
                if tokens_length >= len(content):
                    valid_contents.append((content, tokens_count))
                    break

                # This doc has more than max_tokens: save the head as a separate document and continue
                if tokens_count >= max_tokens:
                    logger.error(
                        "Found unit of text with a token count higher than the maximum allowed. "
                        "The unit is going to be cut at %s tokens, and the remaining %s chars will go to one (or more) new documents. "
                        "Set the maximum amout of tokens allowed through the 'max_tokens' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.",
                        max_tokens,
                        len(content) - tokens_length,
                    )
                    valid_contents.append((content[:tokens_length], tokens_count))
                    content = content[:tokens_length]
                    tokens_count = 0

                # This doc has more than max_chars: save the head as a separate document and continue
                if max_chars and tokens_length >= max_chars:
                    logger.error(
                        "Found unit of text with a character count higher than the maximum allowed. "
                        "The unit is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                        "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.",
                        max_chars,
                        len(content) - max_chars,
                    )
                    valid_contents.append((content[:max_chars], tokens_count))
                    content = content[:max_chars]
                    tokens_count = 0

    # Validate only the chars, split if necessary
    else:
        for content in contents:
            if max_chars and len(content) >= max_chars:
                logger.error(
                    "Found unit of text with a character count higher than the maximum allowed. "
                    "The unit is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                    "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                    "Keep in mind that very long Documents can severely impact the performance of Readers.",
                    max_chars,
                    len(content) - max_chars,
                )
                valid_contents += [
                    (content[max_chars * i : max_chars * (i + 1)], 0) for i in range(int(len(content) / max_chars) + 1)
                ]
            else:
                valid_contents.append((content, 0))

    return valid_contents


def merge_headlines(
    sources: List[Tuple[str, List[Dict[str, Any]]]], separator: str
) -> List[Dict[str, Union[str, int]]]:
    """
    Merges the headlines dictionary with the new position of each headline into the merged document.
    Assumes the documents are in the same order as when they were merged.

    :param sources: tuple (source document content, source document headlines).
    :param separator: the string used to join the document's content
    :return: a dictionary that can be assigned to the merged document's headlines key.
    """
    aligned_headlines = []
    position_in_merged_document = 0
    for content, headlines in sources:
        for headline in headlines:
            headline["start_idx"] += position_in_merged_document
            aligned_headlines.append(headline)
        position_in_merged_document += len(content) + len(separator)
    return aligned_headlines


def common_values(list_of_dicts: List[Dict[str, Any]], exclude: List[str]) -> Dict[str, Any]:
    """
    Retains all keys shared across all the documents being merged.

    Such keys are checked recursively, see tests.

    :param list_of_dicts: dicts to merge
    :param exclude: keys to drop regardless of their content
    :return: the merged dictionary
    """
    shortest_dict = min(list_of_dicts, key=len)
    merge_dictionary = {}
    for key, value in shortest_dict.items():

        # if not all dicts have this key, skip
        if not key in exclude and all(key in dict.keys() for dict in list_of_dicts):

            # if the value is a dictionary, merge recursively
            if isinstance(value, dict):
                list_of_subdicts = [dictionary[key] for dictionary in list_of_dicts]
                merge_dictionary[key] = common_values(list_of_subdicts, exclude=[])

            # If the value is not a dictionary, keep only if the values is the same for all
            elif all(value == dict[key] for dict in list_of_dicts):
                merge_dictionary[key] = value

    return merge_dictionary
