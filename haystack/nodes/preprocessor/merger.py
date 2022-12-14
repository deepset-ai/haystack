from typing import List, Optional, Tuple, Union, Dict, Any, Callable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.modeling.model.feature_extraction import FeatureExtractor
from haystack.nodes.preprocessor.helpers import (
    split_by_separators,
    split_by_transformers_tokenizer,
    validate_unit_boundaries,
    make_merge_groups,
    merge_headlines,
    common_values,
)


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
        tokenizer_model: Optional[
            Union[Literal["word"], Path, PreTrainedTokenizer, FeatureExtractor, Callable]
        ] = "word",
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

        :param tokenizer_model: If `max_tokens` is set, you should provide a tokenizer model to compute the tokens.
                                There are several options, depending on the tradeoff you need between precision and speed:
                                - "word". The text is split with the `split()` function.
                                - A tokenizer model. You can give its identifier on Hugging Face Hub, a local path to load it from,
                                    or an instance of `FeatureExtractor` or Transformer's `PreTrainedTokenizer`.
                                - A lambda function. In this case, make sure it takes one single input parameter called `text`, like
                                  `tokenizer_model=lambda text: text.split("my token delimiter")`

                                Note that a tokenizer is not mandatory as you can also provide the tokens manually to the `merge` method.

                                Defaults to "word".

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

        self._tokenizer = None
        if tokenizer_model or max_tokens:
            self.tokenizer = tokenizer_model

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer_model=Union[str, Path, PreTrainedTokenizer, FeatureExtractor, Callable]):
        if not tokenizer_model:
            raise ValueError(
                "Can't set the tokenizer to None. "
                "Provide either the string 'word', a Hugging Face identifier, a path to a local tokenizer, "
                "or an instance of Haystack's FeatureExtractor or Transformers' PreTrainedTokenizer. "
                "You can also provide your own lambda function to tokenize text: in this case "
                "make sure it takes only one input parameter called 'text'."
            )
        if isinstance(tokenizer_model, (PreTrainedTokenizer, FeatureExtractor)):
            self._tokenizer = lambda text: split_by_transformers_tokenizer(text=text, tokenizer=tokenizer_model)[0]

        elif isinstance(tokenizer_model, (str, Path)):
            if tokenizer_model == "word":
                self._tokenizer = lambda text: split_by_separators(text=text, separators=[" ", "\n", "\f"])
            else:
                self._tokenizer = lambda text: split_by_transformers_tokenizer(
                    text=text, tokenizer=FeatureExtractor(pretrained_model_name_or_path=tokenizer_model)
                )[0]
        else:
            self._tokenizer = tokenizer_model

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

        :param documents: the documents to merge.

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
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        tokens = []
        if max_tokens:
            for document in documents:
                tokens += self.tokenizer(document.content)

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
            tokens=tokens,
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

        :param documents: the documents to merge.

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

        :param documents: the documents to merge.

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

        :param tokens: a single list with all the tokens contained in all documents to merge.
                        So, if the document look like `[Document(content="hello how are you"), Document(content="I'm fine thanks")]`,
                        `tokens` should contain something similar to `["hello", "how", "are", "you", "I'm", "fine", "thanks"]`.

                        If the `tokens` are not given but `max_tokens` is set, `DocumentMerger` will tokenize the text by itself
                        before proceeding.

                        Useful to speed up the token counting process when `max_tokens` is set. Ignored if `max_tokens` is not set.

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

        if max_tokens and not tokens:
            logger.info(
                "DocumentMerger.merge() received `max_tokens` but no `tokens`. Generating the tokens on the fly."
            )
            tokens = []
            for document in documents:
                tokens += self.tokenizer(document.content)

        valid_contents = validate_unit_boundaries(
            contents=[doc.content for doc in documents], max_chars=max_chars, max_tokens=max_tokens, tokens=tokens
        )

        groups_to_merge = make_merge_groups(
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
                pages = [
                    documents[doc_index].meta["page"]
                    for doc_index in group
                    if "page" in documents[doc_index].meta.keys()
                ]
                if pages:
                    merged_documents[group_index].meta["page"] = min(int(page) for page in pages)

        return merged_documents
