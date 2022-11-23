from typing import List, Optional, Union

from pathlib import Path
import logging
import warnings
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes.preprocessor.splitter import DocumentSplitter, SplitBy
from haystack.nodes.preprocessor.cleaner import DocumentCleaner


logger = logging.getLogger(__name__)


REGEX_METACHARS = r".^$*+?{}[]\|()"


class PreProcessor(BaseComponent):

    outgoing_edges = 1

    def __init__(
        self,
        split_by: SplitBy,
        split_length: int,
        clean_whitespace: bool,
        clean_empty_lines: bool,
        clean_header_footer: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
        split_regex: Optional[str] = None,
        split_overlap: int = 0,
        split_max_chars: int = 5000,
        tokenizer_model: Optional[Union[str, Path, PreTrainedTokenizer]] = None,
        nltk_language: str = "english",
        nltk_folder: Optional[str] = None,
        progress_bar: bool = True,
        add_page_number: bool = True,
        clean_substrings: Optional[List[str]] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
    ):
        """
        Perform document cleaning and splitting.

        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param clean_regex: Remove the specified regex matches from the text. For example, `clean_regex='[0-9]'`
                            removes all digits from the document's content, and `clean_regex='(a string|another string)'`
                            will remove all occurrences of either string from the document content.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                    for the longest common string. This heuristic uses exact matches and therefore
                                    works well for footers like "Copyright 2019 by The Author", but won't detect "Page 3 of 4"
                                    or similar. Use 'clean_regex' to detect such headers.
        :param header_footer_n_chars: how many chars to look for headers and footer in. Defautls to 50.
        :param header_footer_pages_to_ignore: which pages to ignore in the header-footer detection heuristic.
                                              Takes a list of pages numbers. It ignores no pages by default.

        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.
        :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document.
                             For instance, if n -> 10 & split_by -> "sentence", then each output document will contain 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                              For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                              `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                                will cut the document, even mid-word, and log a loud error.\n
                                It's recommended to set this value approximately double double the size expect your documents
                                to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                                length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                                This is a safety parameter to avoid extremely long documents to end up in the document store.
                                Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                                performance of Reader nodes and might slow down drastically the indexing speed.

        :param tokenizer_model: If `split_by="token"`, you should provide a tokenizer model to compute the tokens,
                                for example `deepset/roberta-base-squad2`. You can give its identifier on HuggingFace Hub,
                                a local path to load it from, or an instance of PreTrainedTokenizer.
        :param nltk_language: If `split_by="sentence"`, yhe language used by "nltk.tokenize.sent_tokenize", for example "english", or "french".
                              Mind that some languages have limited support by the tokenizer: for example it seems incapable to split Chinese text
                              by word, but it can correctly split it by sentence. Ignored if not `split_by="sentence"`.
        :param nltk_folder: If `split_by="sentence"`, path to the folder containing the NTLK PunktSentenceTokenizer models, if loading a model from a local path.
                            Leave empty otherwise. Ignored if not `split_by="sentence"`.

        :param progress_bar: Whether to show a progress bar.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.

        :param split_respect_sentence_boundary: deprecated, use 'split_by="sentence"' and adjust the value of 'split_length'.
        :param clean_substrings: Deprecated, use 'clean_regex' as 'clean_regex=r"(first substring|second substring|third substring)"'.
        """
        super().__init__()

        if split_respect_sentence_boundary is not None:
            warnings.warn(
                "'split_respect_sentence_boundary' is deprecated. "
                "Setting 'split_by=\"word\"', sentence boundaries are never respected. "
                "Use 'split_by=\"sentence\"' to have the sentence boundaries respected. "
                "However, keep in mind that the 'split_length' will need to be adjusted, "
                "as it now refers to the number of sentences.",
                DeprecationWarning,
            )

        if clean_substrings:
            warnings.warn("clean_substrings is deprecated, use clean_regex", DeprecationWarning)
            clean_regex = f"({'|'.join(clean_substrings)})"

        if split_length <= 0 or not isinstance(split_length, int):
            raise ValueError("split_length must be an integer > 0")

        if split_length:
            if split_overlap < 0 or not isinstance(split_overlap, int):
                raise ValueError("split_overlap must be an integer >= 0")

            if split_overlap >= split_length:
                raise ValueError("split_length must be higher than split_overlap")

        self.splitter = DocumentSplitter(
            split_by=split_by,
            split_regex=split_regex,
            split_length=split_length,
            split_overlap=split_overlap,
            max_chars=split_max_chars,
            tokenizer_model=tokenizer_model,
            nltk_language=nltk_language,
            nltk_folder=nltk_folder,
            progress_bar=progress_bar,
            add_page_number=add_page_number,
        )

        self.cleaner = DocumentCleaner(
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            clean_regex=clean_regex,
            header_footer_n_chars=header_footer_n_chars,
            header_footer_pages_to_ignore=header_footer_pages_to_ignore,
        )
        self.progress_bar = progress_bar

    def run(  # type: ignore
        self,
        documents: List[Document],
        clean_whitespace: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        clean_header_footer: Optional[bool] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
        split_by: Optional[SplitBy] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        add_page_number: Optional[bool] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        clean_substrings: Optional[List[str]] = None,
    ):
        """
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param clean_regex: Remove the specified regex matches from the text. For example, `clean_regex='[0-9]'`
                            removes all digits from the document's content, and `clean_regex='(a string|another string)'`
                            will remove all occurrences of either string from the document content.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by The Author", but won't detect "Page 3 of 4"
                                     or similar.
        :param header_footer_n_chars: how many chars to look for headers and footer in. Defautls to 50.
        :param header_footer_pages_to_ignore: which pages to ignore in the header-footer detection heuristic.
                                              Takes a list of pages numbers. It ignores no pages by default.

        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.
        :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance,
                             if `split_length=10 and split_by="sentence"`, then each output document will have 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                              For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                              `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.

        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.

        :param split_respect_sentence_boundary: deprecated, use 'split_by="sentence"' and adjust the value of 'split_length'.
        :param clean_substrings: Deprecated, use 'clean_regex' as 'clean_regex=r"(first substring|second substring|third substring)"'.
        """
        if split_respect_sentence_boundary is not None:
            warnings.warn(
                "'split_respect_sentence_boundary' is deprecated. "
                "Setting 'split_by=\"word\"', sentence boundaries are never respected. "
                "Use 'split_by=\"sentence\"' to have the sentence boundaries respected. "
                "However, keep in mind that the 'split_length' will need to be adjusted, "
                "as it now refers to the number of sentences.",
                DeprecationWarning,
            )
            self.split_respect_sentence_boundary = split_respect_sentence_boundary

        if clean_substrings:
            warnings.warn("clean_substrings is deprecated, use clean_regex", DeprecationWarning)
            if not clean_regex:
                clean_regex = f"({'|'.join(clean_substrings)})"
            else:
                logger.warning("The value of clean_substrings will be overwritten by the value of clean_regex")

        if isinstance(documents, Document):
            warnings.warn(
                "Passing single documents to Preprocessor.process() is deprecated. Pass a list of Document objects",
                DeprecationWarning,
            )
            documents = [documents]

        elif isinstance(documents, dict):
            warnings.warn(
                "Passing dictionaries to Preprocessor.process() is deprecated. Pass a list of Document objects.",
                DeprecationWarning,
            )
            documents = [Document.from_dict(documents)]

        elif isinstance(documents, list) and isinstance(documents[0], dict):
            warnings.warn(
                "Passing dictionaries to Preprocessor.process() is deprecated. Pass a list of Document objects.",
                DeprecationWarning,
            )
            documents = [Document.from_dict(doc) for doc in documents]

        elif not isinstance(documents, list) or not isinstance(documents[0], Document):
            raise ValueError("'documents' must be a list of Document objects.")

        elif any(document.content_type != "text" for document in documents):
            ids = [doc.id for doc in documents if doc.content_type != "text"]
            raise ValueError(
                "Documents list contains one or more documents that are not of type 'text' "
                f"(doc ids: '{', '.join(ids)}'). Preprocessor only handles text documents."
            )

        documents = self.splitter.run(
            documents=documents,
            split_by=split_by,
            split_regex=split_regex,
            split_length=split_length,
            split_overlap=split_overlap,
            add_page_number=add_page_number,
        )[0]["documents"]

        documents = self.cleaner.run(
            documents=documents,
            clean_whitespace=clean_whitespace,
            clean_empty_lines=clean_empty_lines,
            clean_regex=clean_regex,
            clean_header_footer=clean_header_footer,
            header_footer_n_chars=header_footer_n_chars,
            header_footer_pages_to_ignore=header_footer_pages_to_ignore,
        )[0]["documents"]

        return {"documents": documents}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        clean_whitespace: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        clean_header_footer: Optional[bool] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
        split_by: Optional[SplitBy] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        add_page_number: Optional[bool] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        clean_substrings: Optional[List[str]] = None,
    ):
        nested_docs = [
            self.run(
                documents=docs,
                clean_whitespace=clean_whitespace,
                clean_empty_lines=clean_empty_lines,
                clean_regex=clean_regex,
                clean_header_footer=clean_header_footer,
                header_footer_n_chars=header_footer_n_chars,
                header_footer_pages_to_ignore=header_footer_pages_to_ignore,
                split_by=split_by,
                split_regex=split_regex,
                split_length=split_length,
                split_overlap=split_overlap,
                add_page_number=add_page_number,
                clean_substrings=clean_substrings,
                split_respect_sentence_boundary=split_respect_sentence_boundary,
            )
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Preprocessing", unit="docs")
        ]
        return {"documents": nested_docs}, "output_1"
