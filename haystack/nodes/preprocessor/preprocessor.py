import logging
import re
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import List, Optional, Generator, Set, Union, Tuple, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import warnings
from pathlib import Path
from pickle import UnpicklingError

import nltk
from more_itertools import windowed
from tqdm.auto import tqdm

from haystack.nodes.preprocessor.base import BasePreProcessor
from haystack.errors import HaystackError
from haystack.schema import Document


logger = logging.getLogger(__name__)


iso639_to_nltk = {
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ml": "malayalam",
}


class PreProcessor(BasePreProcessor):
    def __init__(
        self,
        clean_whitespace: bool = True,
        clean_header_footer: bool = False,
        n_chars: int = 300,
        n_first_pages_to_ignore: int = 1,
        n_last_pages_to_ignore: int = 1,
        clean_empty_lines: bool = True,
        remove_substrings: List[str] = [],
        remove_numeric_tables: bool = True,
        pre_split_paragraphs: bool = False,
        split_by: Literal["word", "sentence", "passage", None] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_respect_sentence_boundary: bool = True,
        language: str = "en",
        tokenizer_model_folder: Optional[Union[str, Path]] = None,
        id_hash_keys: Optional[List[str]] = None,
        progress_bar: bool = True,
        add_page_number: bool = False,
        merge_short: int = 10,
        merge_lowercase: bool = True,
        remove_linebreaks: bool = True,
    ):
        """
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param n_chars: Number of characters to consider for the header and footer removal heuristic.
        :param n_first_pages_to_ignore: number of first pages to ignore when removing(e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param remove_substrings: Remove specified substrings from the text.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param pre_split_paragraphs: Whether to split the text into paragraphs before splitting it into smaller chunks.
                            This is useful if you want the chunking to only happen within paragraphs, so as to maintain maximum context for vector embeddings
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        :param language: The language used by "nltk.tokenize.sent_tokenize" in iso639 format.
            Available options: "ru","sl","es","sv","tr","cs","da","nl","en","et","fi","fr","de","el","it","no","pl","pt","ml"
        :param tokenizer_model_folder: Path to the folder containing the NTLK PunktSentenceTokenizer models, if loading a model from a local path. Leave empty otherwise.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param progress_bar: Whether to show a progress bar.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`. If a doc's text spans multiple pages, the page number of the
                                first page is added to the meta field.
        :param merge_short: Whether to merge short paragraphs into the previous paragraph. This is useful for PDFs
                            where paragraphs are split across pages and the last paragraph of a page is very short.
                            Set to a number of characters to merge paragraphs shorter than that number. Default is 10 characters, set to 0 to disable.
        :param merge_lowercase: Whether to merge paragraphs that start with a lowercase letter into the previous
                                paragraph. This is useful for PDFs where paragraphs are split across pages and the
                                first paragraph of a page starts with a lowercase letter.
        :param remove_linebreaks: Whether to remove line breaks from the text. This is useful for PDFs where
                                      paragraphs are split across pages and the last paragraph of a page ends with a
                                        line break.
        """
        super().__init__()

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        self.clean_whitespace = clean_whitespace
        self.clean_header_footer = clean_header_footer
        self.clean_empty_lines = clean_empty_lines
        self.remove_substrings = remove_substrings
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_respect_sentence_boundary = split_respect_sentence_boundary
        self.language = language
        self.tokenizer_model_folder = tokenizer_model_folder
        self.id_hash_keys = id_hash_keys
        self.progress_bar = progress_bar
        self.add_page_number = add_page_number
        self.pre_split_paragraphs = pre_split_paragraphs
        self.remove_numeric_tables = remove_numeric_tables
        self.n_chars = n_chars
        self.n_first_pages_to_ignore = n_first_pages_to_ignore
        self.n_last_pages_to_ignore = n_last_pages_to_ignore
        self.merge_short = merge_short
        self.merge_lowercase = merge_lowercase
        self.remove_linebreaks = remove_linebreaks

    def process(
        self,
        documents: Union[dict, Document, List[Union[dict, Document]]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        remove_substrings: List[str] = [],
        split_by: Literal["word", "sentence", "passage", None] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
        pre_split_paragraphs: Optional[bool] = None,
        remove_numeric_tables: Optional[bool] = None,
        n_chars: Optional[int] = None,
        n_first_pages_to_ignore: Optional[int] = None,
        n_last_pages_to_ignore: Optional[int] = None,
        merge_short: Optional[int] = None,
        merge_lowercase: Optional[bool] = None,
        remove_linebreaks: Optional[bool] = None,
    ) -> List[Document]:

        """
        Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.

        :param documents: A single document or a list of documents to be cleaned and split.
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param remove_substrings: Remove specified substrings from the text.
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param pre_split_paragraphs: Whether to split the text into paragraphs before splitting it into smaller chunks.
                            This is useful if you want the chunking to only happen within paragraphs, so as to maintain maximum context for vector embeddings
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param n_chars: Number of characters to consider for the header and footer removal heuristic.
        :param n_first_pages_to_ignore: number of first pages to ignore when removing(e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :param merge_short: Whether to merge short paragraphs into the previous paragraph. This is useful for PDFs
                            where paragraphs are split across pages and the last paragraph of a page is very short.
                            Set to a number of characters to merge paragraphs shorter than that number. Default is 10 characters, set to 0 to disable.
        :param merge_lowercase: Whether to merge paragraphs that start with a lowercase letter into the previous
                                paragraph. This is useful for PDFs where paragraphs are split across pages and the
                                first paragraph of a page starts with a lowercase letter.
        :param remove_linebreaks: Whether to remove line breaks from the text. This is useful for PDFs where
                                      paragraphs are split across pages and the last paragraph of a page ends with a
                                        line break.
        """
        if not isinstance(documents, list):
            warnings.warn(
                "Using a single Document as argument to the 'documents' parameter is deprecated. Use a list "
                "of (a single) Document instead.",
                DeprecationWarning,
                2,
            )

        kwargs = {
            "clean_whitespace": clean_whitespace,
            "clean_header_footer": clean_header_footer,
            "clean_empty_lines": clean_empty_lines,
            "remove_substrings": remove_substrings,
            "split_by": split_by,
            "split_length": split_length,
            "split_overlap": split_overlap,
            "split_respect_sentence_boundary": split_respect_sentence_boundary,
            "pre_split_paragraphs": pre_split_paragraphs,
            "remove_numeric_tables": remove_numeric_tables,
            "n_chars": n_chars,
            "n_first_pages_to_ignore": n_first_pages_to_ignore,
            "n_last_pages_to_ignore": n_last_pages_to_ignore,
            "merge_short": merge_short,
            "merge_lowercase": merge_lowercase,
            "remove_linebreaks": remove_linebreaks,
        }

        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(documents, (Document, dict)):
            ret = self._process_single(document=documents, id_hash_keys=id_hash_keys, **kwargs)  # type: ignore
        elif isinstance(documents, list):
            ret = self._process_batch(documents=list(documents), id_hash_keys=id_hash_keys, **kwargs)
        else:
            raise Exception("documents provided to PreProcessor.prepreprocess() is not of type list nor Document")

        return ret

    def _process_single(
        self,
        document: Union[dict, Document],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        remove_substrings: List[str] = [],
        split_by: Literal["word", "sentence", "passage", None] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
        pre_split_paragraphs: Optional[bool] = None,
        remove_numeric_tables: Optional[bool] = None,
        n_chars: Optional[int] = None,
        n_first_pages_to_ignore: Optional[int] = None,
        n_last_pages_to_ignore: Optional[int] = None,
        merge_short: Optional[int] = None,
        merge_lowercase: Optional[bool] = None,
        remove_linebreaks: Optional[bool] = None,
        add_page_number: Optional[bool] = None,
    ) -> List[Document]:
        """
        Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.

        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param remove_substrings: Remove specified substrings from the text.
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param pre_split_paragraphs: Whether to split the text into paragraphs before splitting it into smaller chunks.
                            This is useful if you want the chunking to only happen within paragraphs, so as to maintain maximum context for vector embeddings
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param n_chars: Number of characters to consider for the header and footer removal heuristic.
        :param n_first_pages_to_ignore: number of first pages to ignore when removing(e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :param merge_short: Whether to merge short paragraphs into the previous paragraph. This is useful for PDFs
                            where paragraphs are split across pages and the last paragraph of a page is very short.
                            Set to a number of characters to merge paragraphs shorter than that number. Default is 10 characters, set to 0 to disable.
        :param merge_lowercase: Whether to merge paragraphs that start with a lowercase letter into the previous
                                paragraph. This is useful for PDFs where paragraphs are split across pages and the
                                first paragraph of a page starts with a lowercase letter.
        :param remove_linebreaks: Whether to remove line breaks from the text. This is useful for PDFs where
                                      paragraphs are split across pages and the last paragraph of a page ends with a
                                        line break.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`. If a doc's text spans multiple pages, the page number of the
                                first page is added to the meta field.
        :return: List of processed documents
        """
        if clean_whitespace is None:
            clean_whitespace = self.clean_whitespace
        if clean_header_footer is None:
            clean_header_footer = self.clean_header_footer
        if clean_empty_lines is None:
            clean_empty_lines = self.clean_empty_lines
        if not remove_substrings:
            remove_substrings = self.remove_substrings
        if split_by is None:
            split_by = self.split_by
        if split_length is None:
            split_length = self.split_length
        if split_overlap is None:
            split_overlap = self.split_overlap
        if split_respect_sentence_boundary is None:
            split_respect_sentence_boundary = self.split_respect_sentence_boundary
        if pre_split_paragraphs is None:
            pre_split_paragraphs = self.pre_split_paragraphs
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if n_chars is None:
            n_chars = self.n_chars
        if n_first_pages_to_ignore is None:
            n_first_pages_to_ignore = self.n_first_pages_to_ignore
        if n_last_pages_to_ignore is None:
            n_last_pages_to_ignore = self.n_last_pages_to_ignore
        if merge_short is None:
            merge_short = self.merge_short
        if merge_lowercase is None:
            merge_lowercase = self.merge_lowercase
        if remove_linebreaks is None:
            remove_linebreaks = self.remove_linebreaks
        if add_page_number is None:
            add_page_number = self.add_page_number

        cleaned_document = self.clean(
            document=document,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            remove_substrings=remove_substrings,
            id_hash_keys=id_hash_keys,
            remove_numeric_tables=remove_numeric_tables,
            n_chars=n_chars,
            n_first_pages_to_ignore=n_first_pages_to_ignore,
            n_last_pages_to_ignore=n_last_pages_to_ignore,
            remove_linebreaks=remove_linebreaks,
        )
        split_documents = self.split(
            document=cleaned_document,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
            id_hash_keys=id_hash_keys,
            pre_split_paragraphs=pre_split_paragraphs,
            merge_short=merge_short,
            merge_lowercase=merge_lowercase,
            add_page_number=add_page_number,
        )
        return split_documents

    def _process_batch(
        self, documents: List[Union[dict, Document]], id_hash_keys: Optional[List[str]] = None, **kwargs
    ) -> List[Document]:
        """

        :param documents: List of documents to be processed.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :return: List of processed documents.
        """
        nested_docs = [
            self._process_single(d, id_hash_keys=id_hash_keys, **kwargs)
            for d in tqdm(documents, disable=not self.progress_bar, desc="Preprocessing", unit="docs")
        ]
        return [d for x in nested_docs for d in x]

    def clean(
        self,
        document: Union[dict, Document],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        remove_substrings: List[str],
        id_hash_keys: Optional[List[str]] = None,
        remove_linebreaks: Optional[bool] = True,
        remove_numeric_tables: Optional[bool] = None,
        n_chars: Optional[int] = 300,
        n_first_pages_to_ignore: Optional[int] = 1,
        n_last_pages_to_ignore: Optional[int] = 1,
    ) -> Document:
        """
        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().

        :param document: Document to clean
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param remove_substrings: Remove specified substrings from the text.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param remove_linebreaks: Whether to remove line breaks from the text. This is useful for PDFs where
                                      paragraphs are split across pages and the last paragraph of a page ends with a
                                        line break.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param n_chars: Number of characters to consider for the header and footer removal heuristic.
        :param n_first_pages_to_ignore: number of first pages to ignore when removing(e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        """
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(document, dict):
            document = Document.from_dict(document, id_hash_keys=id_hash_keys)

        # Mainly needed for type checking
        if not isinstance(document, Document):
            raise HaystackError("Document must not be of type 'dict' but of type 'Document'.")

        if type(document.content) is not str:
            logger.error("Document content is not of type str. Nothing to clean.")
            return document

        text = document.content
        if clean_header_footer:
            text = self._find_and_remove_header_footer(
                text,
                n_chars=n_chars,
                n_first_pages_to_ignore=n_first_pages_to_ignore,
                n_last_pages_to_ignore=n_last_pages_to_ignore,
            )

        pages = text.split("\f")
        cleaned_pages = []
        for i, page in enumerate(pages):
            if not page:
                # there are many "empty text" pages in a marketing document, as for example the cover page. If we just forget about them, we have a mismatch
                # with page numbers which causes problems later on. Therefore, we replace them with a dummy text, which will not be found by any query.
                page = EMPTY_PAGE_PLACEHOLDER
            lines = page.splitlines()

            # Removes empty lines from the start of a text so as to not generate empty paragraphs
            line_len = 0
            if clean_empty_lines and i == 0:
                while line_len == 0:
                    line_len = len(lines[0].strip())
                    if line_len == 0:
                        del lines[0]

            cleaned_lines = []
            for line in lines:
                if clean_whitespace:
                    line = line.strip()

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    words = line.split()
                    digits = [word for word in words if any(i.isdigit() for i in word)]
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        continue
                cleaned_lines.append(line)
            cleaned_page = "\n".join(cleaned_lines)
            cleaned_pages.append(cleaned_page)

        text = "\f".join(cleaned_pages)

        # Tidy up linebreaks that were generated through the extraction and cleaning process - merge a hyphenated word that spans two lines and separate the rest by spaces.
        if remove_linebreaks:
            text_shards = text.split("\n\n")
            text_shards = [t.replace("-\n", "").replace("\n", " ") for t in text_shards]
            text = "\n\n".join(text_shards)

        # remove any excess empty lines after cleaning up the line breaks
        if clean_empty_lines:
            text, headlines = self._clean_empty_lines(text=text, headlines=headlines)

        for substring in remove_substrings:
            text, headline = self._remove_substring(text=text, substring=substring, headlines=headlines)

        if text != document.content:
            document = deepcopy(document)
            document.content = text
        if headlines:
            document.meta["headlines"] = headlines

        return document

    def split(
        self,
        document: Union[dict, Document],
        split_by: Literal["word", "sentence", "passage", None],
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool,
        merge_short: Optional[int] = 10,
        merge_lowercase: Optional[bool] = True,
        pre_split_paragraphs: Optional[bool] = False,
        id_hash_keys: Optional[List[str]] = None,
        add_page_number: Optional[bool] = False,
    ) -> List[Document]:
        """Perform document splitting on a single document. This method can split on different units, at different lengths,
        with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
        the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents.

        :param document: Document to split
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage".
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        :param merge_short: Whether to merge short paragraphs into the previous paragraph. This is useful for PDFs
                            where paragraphs are split across pages and the last paragraph of a page is very short.
                            Set to a number of characters to merge paragraphs shorter than that number. Default is 10 characters, set to 0 to disable.
        :param merge_lowercase: Whether to merge paragraphs that start with a lowercase letter into the previous
                                paragraph. This is useful for PDFs where paragraphs are split across pages and the
                                first paragraph of a page starts with a lowercase letter.
        :param pre_split_paragraphs: Whether to split the text into paragraphs before splitting it into smaller chunks.
                            This is useful if you want the chunking to only happen within paragraphs, so as to maintain maximum context for vector embeddings
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`. If a doc's text spans multiple pages, the page number of the
                                first page is added to the meta field.
        :return: List of documents
        """
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(document, dict):
            document = Document.from_dict(document, id_hash_keys=id_hash_keys)
        # Mainly needed for type checking
        if not isinstance(document, Document):
            raise HaystackError("Document must not be of type 'dict' but of type 'Document'.")

        if type(document.content) is not str:
            logger.error("Document content is not of type str. Nothing to split.")
            return [document]

        if not split_by:
            return [document]

        if pre_split_paragraphs and split_by == "passage":
            raise ValueError('"pre_split_paragraphs=True" is not compatible with split_by="passage"')

        if not split_by in ["word", "sentence", "passage"]:
            raise ValueError(f'split_by must be one of: "word", "sentence", "passage", not "{split_by}"')

        if not split_length:
            raise ValueError("split_length needs be set when using split_by.")

        if split_overlap >= split_length:
            raise ValueError("split_length must be greater than split_overlap")

        if split_respect_sentence_boundary and split_by != "word":
            raise ValueError("'split_respect_sentence_boundary=True' is only compatible with split_by='word'.")

        text = document.content
        headlines = document.meta["headlines"] if "headlines" in document.meta else []

        # Split by paragraph/passage first, if possible. This allows for maximum context when creating embeddings. Mechanism contained in separate private method so that it can be used later if pre_split_paragraphs is set to False
        if pre_split_paragraphs:
            paras = self._split_paragraphs(text, merge_short, merge_lowercase)
        else:
            paras = [text]

        # split by words ensuring no sub sentence splits
        if split_respect_sentence_boundary and split_by == "word":
            cur_page = 1
            splits_pages = []
            text_splits = []

            # added for loop so that we can split by paragraph/passage first, and then by sentence-constrained words within each paragraph/passage. If pre_split_paragraphs is set to False, then this for loop will only run once
            for para in paras:
                if add_page_number:
                    # SentenceTokenizer will remove "\f" if it is at the end of a sentence, so substituting it in these
                    # cases for "[NEW_PAGE]" to don't lose any page breaks.
                    para = self._substitute_page_breaks(para)
                sentences = self._split_sentences(para)

                sentence_buffer = ""
                for i, sentence in enumerate(sentences):

                    # if pre_split_paragraph wasn't run, need to clean up the \n\n breaks that were left behind by clean() to split paragraphs
                    if len(paras) == 1:
                        sentence = sentence.replace("\n\n", " ").strip()

                    if add_page_number:
                        sentence = sentence.replace("[NEW_PAGE]", "\f")
                        cur_page += sentence.count("\f")

                    # Short or first sentence: put in buffer and continue
                    if len(sentence_buffer.split() + sentence.split()) <= split_length or i == 0:
                        sentence_buffer += " " + sentence
                        splits_pages.append(cur_page)
                        continue

                    # Buffer is full: empty it
                    if len(sentence_buffer.split()) > split_length:
                        # Case of an overfull buffer (can happen with big split_lenght/split_overlap ratios)
                        logger.warning(
                            "Found sentence with a word count higher than the split length. "
                            "This can cause latency problems with your Reader. "
                            "The piece (including the overlap buffer) is %s words, %s chars long. First 20 chars: %s",
                            len(sentence_buffer.split()),
                            len(sentence_buffer),
                            sentence_buffer[:20],
                        )
                    text_splits.append(sentence_buffer.strip())
                    splits_pages.append(cur_page)

                    # Compute the overlap and overwrite the buffer
                    if split_overlap:
                        sentence_buffer = " ".join(sentence_buffer.split()[-split_overlap:])
                    else:
                        sentence_buffer = ""

                    # Add new sentence to buffer
                    sentence_buffer += " " + sentence

                if sentence_buffer:
                    text_splits.append(sentence_buffer.strip())
                    splits_pages.append(cur_page)
        else:
            # create individual "elements" of passage, sentence, or word
            text_splits = []
            splits_pages = []
            cur_page = 1
            # Loop through each paragraph/passage. If pre_split_paragraphs is set to False, then this for loop will only run once
            for para in paras:
                if split_by == "passage":
                    elements = self._split_paragraphs(para, merge_short, merge_lowercase)
                elif split_by == "sentence":
                    if add_page_number:
                        # SentenceTokenizer will remove "\f" if it is at the end of a sentence, so substituting it in these
                        # cases for "[NEW_PAGE]" to don't lose any page breaks.
                        para = self._substitute_page_breaks(para)
                    elements = self._split_sentences(para)

                    for i, sentence in enumerate(elements):
                        if len(paras) == 1:
                            elements[i] = sentence.replace("\n\n", " ")
                        if add_page_number and sentence.startswith("[NEW_PAGE]"):
                            elements[i] = sentence.replace("[NEW_PAGE]", "\f")
                elif split_by == "word":
                    elements = para.replace("\n\n", " ").strip().split(" ")
                else:
                    raise ValueError("PreProcessor only supports 'passage', 'sentence' or 'word' split_by options.")

                # concatenate individual elements based on split_length & split_stride

                if split_overlap:
                    segments = windowed(elements, n=split_length, step=split_length - split_overlap)
                else:
                    segments = windowed(elements, n=split_length, step=split_length)

                for seg in segments:
                    current_units = [unit for unit in seg if unit is not None]
                    txt = " ".join(current_units)
                    if len(txt) > 0:
                        text_splits.append(txt)
                        splits_pages.append(cur_page)
                        if add_page_number:
                            processed_units = current_units[: split_length - split_overlap]
                            num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)
                            cur_page += num_page_breaks

        # create new document dicts for each text split
        documents = self._create_docs_from_splits(
            text_splits=text_splits,
            splits_pages=splits_pages,
            splits_start_idxs=splits_start_idxs,
            headlines=headlines,
            meta=document.meta or {},
            id_hash_keys=id_hash_keys,
        )

        return documents

    @staticmethod
    def _clean_whitespace(text: str, headlines: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Strips whitespaces before or after each line in the text.
        """
        pages = text.split("\f")
        cleaned_pages = []
        cur_headline_idx = 0
        num_headlines = len(headlines)
        cur_char_idx = 0
        num_removed_chars_total = 0
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for idx, line in enumerate(lines):
                old_line_len = len(line)
                cleaned_line = line.strip()
                cleaned_line_len = len(cleaned_line)
                cur_char_idx += old_line_len + 1  # add 1 for newline char
                if old_line_len != cleaned_line_len:
                    num_removed_chars_current = old_line_len - cleaned_line_len
                    num_removed_chars_total += num_removed_chars_current
                    for headline_idx in range(cur_headline_idx, num_headlines):
                        if cur_char_idx - num_removed_chars_total <= headlines[headline_idx]["start_idx"]:
                            headlines[headline_idx]["start_idx"] -= num_removed_chars_current
                        else:
                            cur_headline_idx += 1

                cleaned_lines.append(cleaned_line)
            cleaned_page = "\n".join(cleaned_lines)
            cleaned_pages.append(cleaned_page)

        cleaned_text = "\f".join(cleaned_pages)
        return cleaned_text, headlines

    @staticmethod
    def _clean_empty_lines(text: str, headlines: List[Dict]) -> Tuple[str, List[Dict]]:
        if headlines:
            num_headlines = len(headlines)
            multiple_new_line_matches = re.finditer(r"\n\n\n+", text)
            cur_headline_idx = 0
            num_removed_chars_accumulated = 0
            for match in multiple_new_line_matches:
                num_removed_chars_current = match.end() - match.start() - 2
                for headline_idx in range(cur_headline_idx, num_headlines):
                    if match.end() - num_removed_chars_accumulated <= headlines[headline_idx]["start_idx"]:
                        headlines[headline_idx]["start_idx"] -= num_removed_chars_current
                    else:
                        cur_headline_idx += 1
                num_removed_chars_accumulated += num_removed_chars_current

        cleaned_text = re.sub(r"\n\n\n+", "\n\n", text)
        return cleaned_text, headlines

    @staticmethod
    def _remove_substring(text: str, substring: str, headlines: List[Dict]) -> Tuple[str, List[Dict]]:
        if headlines:
            num_headlines = len(headlines)
            multiple_substring_matches = re.finditer(substring, text)
            cur_headline_idx = 0
            num_removed_chars_accumulated = 0
            for match in multiple_substring_matches:
                for headline_idx in range(cur_headline_idx, num_headlines):
                    if match.end() - num_removed_chars_accumulated <= headlines[headline_idx]["start_idx"]:
                        headlines[headline_idx]["start_idx"] -= len(substring)
                    else:
                        cur_headline_idx += 1
                num_removed_chars_accumulated += len(substring)

        cleaned_text = text.replace(substring, "")
        return cleaned_text, headlines

    def _split_by_word_respecting_sent_boundary(
        self, text: str, split_length: int, split_overlap: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Splits the text into parts of split_length words while respecting sentence boundaries.
        """
        sentences = self._split_sentences(text)

        word_count_slice = 0
        cur_page = 1
        cur_start_idx = 0
        splits_pages = []
        list_splits = []
        splits_start_idxs = []
        current_slice: List[str] = []
        for sen in sentences:
            word_count_sen = len(sen.split())

            if word_count_sen > split_length:
                long_sentence_message = (
                    f"We found one or more sentences whose word count is higher than the split length."
                )
                if long_sentence_message not in self.print_log:
                    self.print_log.add(long_sentence_message)
                    logger.warning(long_sentence_message)

            if word_count_slice + word_count_sen > split_length:
                # Number of words exceeds split_length -> save current slice and start a new one
                if current_slice:
                    list_splits.append(current_slice)
                    splits_pages.append(cur_page)
                    splits_start_idxs.append(cur_start_idx)

                if split_overlap:
                    overlap = []
                    processed_sents = []
                    word_count_overlap = 0
                    current_slice_copy = deepcopy(current_slice)
                    for idx, s in reversed(list(enumerate(current_slice))):
                        sen_len = len(s.split())
                        if word_count_overlap < split_overlap:
                            overlap.append(s)
                            word_count_overlap += sen_len
                            current_slice_copy.pop(idx)
                        else:
                            processed_sents = current_slice_copy
                            break
                    current_slice = list(reversed(overlap))
                    word_count_slice = word_count_overlap
                else:
                    processed_sents = current_slice
                    current_slice = []
                    word_count_slice = 0

                cur_start_idx += len("".join(processed_sents))

                # Count number of page breaks in processed sentences
                if self.add_page_number:
                    num_page_breaks = self._count_processed_page_breaks(
                        sentences=processed_sents,
                        split_overlap=split_overlap,
                        overlapping_sents=current_slice,
                        current_sent=sen,
                    )
                    cur_page += num_page_breaks

            current_slice.append(sen)
            word_count_slice += word_count_sen

        if current_slice:
            list_splits.append(current_slice)
            splits_pages.append(cur_page)
            splits_start_idxs.append(cur_start_idx)

        text_splits = []
        for sl in list_splits:
            txt = "".join(sl)
            if len(txt) > 0:
                text_splits.append(txt)

        return text_splits, splits_pages, splits_start_idxs

    def _split_into_units(self, text: str, split_by: str) -> Tuple[List[str], str]:
        if split_by == "passage":
            elements = text.split("\n\n")
            split_at = "\n\n"
        elif split_by == "sentence":
            elements = self._split_sentences(text)
            split_at = ""  # whitespace will be preserved while splitting text into sentences
        elif split_by == "word":
            elements = text.split(" ")
            split_at = " "
        else:
            raise NotImplementedError("PreProcessor only supports 'passage', 'sentence' or 'word' split_by options.")

        return elements, split_at

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_at: str
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Concatenates the elements into parts of split_length units.
        """
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        split_at_len = len(split_at)
        text_splits = []
        splits_pages = []
        splits_start_idxs = []
        cur_page = 1
        cur_start_idx = 0
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = split_at.join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)
                processed_units = current_units[: split_length - split_overlap]
                cur_start_idx += len((split_at_len * " ").join(processed_units)) + split_at_len
                if self.add_page_number:
                    num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)
                    cur_page += num_page_breaks

        return text_splits, splits_pages, splits_start_idxs

    def _create_docs_from_splits(
        self,
        text_splits: List[str],
        splits_pages: List[int],
        splits_start_idxs: List[int],
        headlines: List[Dict],
        meta: Dict,
        id_hash_keys=Optional[List[str]],
    ) -> List[Document]:
        """
        Creates Document objects from text splits enriching them with page number and headline information if given.
        """
        documents = []

        earliest_rel_hl = 0
        for i, txt in enumerate(text_splits):
            meta = deepcopy(meta)
            doc = Document(content=txt, meta=meta, id_hash_keys=id_hash_keys)
            doc.meta["_split_id"] = i
            if add_page_number:
                doc.meta["page"] = splits_pages[i]
            if headlines:
                split_start_idx = splits_start_idxs[i]
                relevant_headlines, earliest_rel_hl = self._extract_relevant_headlines_for_split(
                    headlines=headlines, split_txt=txt, split_start_idx=split_start_idx, earliest_rel_hl=earliest_rel_hl
                )
                doc.meta["headlines"] = relevant_headlines

            documents.append(doc)

        return documents

    def _split_paragraphs(self, text: str, merge_short: int = 10, merge_lowercase: bool = True) -> List:
        """
        Mechanism to split text into paragraphs, merging paragraphs that are short or span two pages, cleaning up the text in the process.

        param: text: document text to split
        param: merge_short: int, Number of characters below which a paragraph will be merged into the prior paragraph. 10 is the default, set to 0 to disable.
        param: merge_lowercase: bool indicating whether to merge paragraphs that are lowercase (spanning two pages)
        return: list of paragraphs
        """
        paras = text.split("\n\n")

        # Join short paragraphs and paragraphs that span pages
        if paras:
            paras_new = []
            last_para = ""
            for para in paras:
                if not para:
                    continue
                para = para.strip()
                # this paragraph is less than the value of merge_short (default 10 characters) characters or 2 words
                para_is_short = len(para) < merge_short or len(para.split(" ")) < 2
                # this paragraph starts with a lower case and last paragraph does not end with a punctuation
                para_is_lowercase = para and para[0].islower() and last_para and last_para[-1] not in r'.?!"\'\]\)'

                # merge paragraphs to improve qa
                if (merge_short > 0 and para_is_short) or (merge_lowercase and para_is_lowercase):
                    last_para += " " + para
                else:
                    if last_para:
                        paras_new.append(last_para.strip())
                    last_para = para
            # don't forget the last one
            if last_para:
                paras_new.append(last_para.strip())
            paras = paras_new
        return paras

    def _find_and_remove_header_footer(
        self, text: str, n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
    ) -> str:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common string.
        For headers we only search in the first n_chars characters (for footer: last n_chars).
        Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
         but won't detect "Page 3 of 4" or similar.

        :param n_chars: number of first/last characters where the header/footer shall be searched in
        :param n_first_pages_to_ignore: number of first pages to ignore (e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :return: (cleaned pages, found_header_str, found_footer_str)
        """

        pages = text.split("\f")

        # header
        start_of_pages = [p[:n_chars] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
        found_header = self._find_longest_common_ngram(start_of_pages)
        if found_header:
            pages = [page.replace(found_header, "") for page in pages]

        # footer
        end_of_pages = [p[-n_chars:] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
        found_footer = self._find_longest_common_ngram(end_of_pages)
        if found_footer:
            pages = [page.replace(found_footer, "") for page in pages]
        logger.debug("Removed header '%s' and footer '%s' in document", found_header, found_footer)
        text = "\f".join(pages)
        return text

    def _ngram(self, seq: str, n: int) -> Generator[str, None, None]:
        """
        Return ngram (of tokens - currently split by whitespace)

        :param seq: str, string from which the ngram shall be created
        :param n: int, n of ngram
        :return: str, ngram as string
        """

        # In order to maintain the original whitespace, but still consider \n and \t for n-gram tokenization,
        # we add a space here and remove it after creation of the ngrams again (see below)
        seq = seq.replace("\n", " \n")
        seq = seq.replace("\t", " \t")

        words = seq.split(" ")
        ngrams = (
            " ".join(words[i : i + n]).replace(" \n", "\n").replace(" \t", "\t") for i in range(0, len(words) - n + 1)
        )

        return ngrams

    def _allngram(self, seq: str, min_ngram: int, max_ngram: int) -> Set[str]:
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        res = set(chain.from_iterable(ngrams))
        return res

    def _find_longest_common_ngram(
        self, sequences: List[str], max_ngram: int = 30, min_ngram: int = 3
    ) -> Optional[str]:
        """
        Find the longest common ngram across different text sequences (e.g. start of pages).
        Considering all ngrams between the specified range. Helpful for finding footers, headers etc.

        :param sequences: list[str], list of strings that shall be searched for common n_grams
        :param max_ngram: int, maximum length of ngram to consider
        :param min_ngram: minimum length of ngram to consider
        :return: str, common string of all sections
        """
        sequences = [s for s in sequences if s]  # filter empty sequences
        if not sequences:
            return None
        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)

        try:
            longest = max(intersection, key=len)
        except ValueError:
            # no common sequence found
            longest = ""
        return longest if longest.strip() else None

    def _split_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.

        :param text: str, text to tokenize
        :return: list[str], list of sentences
        """
        language_name = iso639_to_nltk.get(self.language)

        sentence_tokenizer = self._load_sentence_tokenizer(language_name)
        # The following adjustment of PunktSentenceTokenizer is inspired by:
        # https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
        # It is needed for preserving whitespace while splitting text into sentences.
        period_context_fmt = r"""
            %(SentEndChars)s             # a potential sentence ending
            \s*                          # match potential whitespace (is originally in lookahead assertion)
            (?=(?P<after_tok>
                %(NonWord)s              # either other punctuation
                |
                (?P<next_tok>\S+)        # or some other token - original version: \s+(?P<next_tok>\S+)
            ))"""
        re_period_context = re.compile(
            period_context_fmt
            % {
                "NonWord": sentence_tokenizer._lang_vars._re_non_word_chars,
                "SentEndChars": sentence_tokenizer._lang_vars._re_sent_end_chars,
            },
            re.UNICODE | re.VERBOSE,
        )
        sentence_tokenizer._lang_vars._re_period_context = re_period_context

        sentences = sentence_tokenizer.tokenize(text)
        return sentences

    def _load_sentence_tokenizer(self, language_name: Optional[str]) -> nltk.tokenize.punkt.PunktSentenceTokenizer:

        # Try to load a custom model from 'tokenizer_model_path'
        if self.tokenizer_model_folder is not None:
            tokenizer_model_path = Path(self.tokenizer_model_folder).absolute() / f"{self.language}.pickle"
            try:
                sentence_tokenizer = nltk.data.load(f"file:{str(tokenizer_model_path)}", format="pickle")
            except (LookupError, UnpicklingError, ValueError) as e:
                if isinstance(e, LookupError):
                    logger.exception(f"PreProcessor couldn't load sentence tokenizer from %s", tokenizer_model_path)
                else:
                    logger.exception(
                        f"PreProcessor couldn't determine model format of sentence tokenizer at %s",
                        tokenizer_model_path,
                    )

                # NLTK failed to load custom SentenceTokenizer, fallback to the default model or to English
                if language_name is not None:
                    logger.error(
                        f"PreProcessor couldn't find custom sentence tokenizer model for {self.language}. "
                        f"Using default {self.language} model."
                    )
                    sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/{language_name}.pickle")
                else:
                    logger.error(
                        f"PreProcessor couldn't find default or custom sentence tokenizer model for {self.language}. "
                        f"Using English instead."
                    )
                    sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/english.pickle")

        # Use a default NLTK model
        elif language_name is not None:
            sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/{language_name}.pickle")
        else:
            logger.error(
                f"PreProcessor couldn't find the default sentence tokenizer model for {self.language}. "
                f" Using English instead. You may train your own model and use the 'tokenizer_model_folder' parameter."
            )
            sentence_tokenizer = nltk.data.load(f"tokenizers/punkt/english.pickle")

        return sentence_tokenizer

    @staticmethod
    def _count_processed_page_breaks(
        sentences: List[str], split_overlap: int, overlapping_sents: List[str], current_sent: str
    ) -> int:
        """
        Counts the number of processed page breaks in a list of processed sentences.

        :param sentences: list[str], sentences to count page breaks in
        :param split_overlap: int, number of sentences to overlap
        :param overlapping_sents: list[str], sentences that overlap with the current sentence
        :param current_sent: str, current sentence
        :return: int, number of processed page breaks
        """
        num_page_breaks = sum(sent.count("\f") for sent in sentences)
        if sentences and sentences[0].startswith("\f"):
            # Remove already used page break
            num_page_breaks -= 1
        # Increment page counter if new split starts with a page break
        if split_overlap and overlapping_sents:
            if overlapping_sents[0].startswith("\f"):
                num_page_breaks += 1
        else:
            if current_sent.startswith("\f"):
                num_page_breaks += 1

        return num_page_breaks

    @staticmethod
    def _substitute_page_breaks(text: str) -> str:
        """
        This method substitutes the page break character "\f" for "[NEW_PAGE]" if it is at the end of a sentence.

        :param text: str, text to substitute page breaks in
        """
        # This regex matches any of sentence-ending punctuation (one of ".", ":", "?", "!") followed by a page break
        # character ("\f") and replaces the page break character with "[NEW_PAGE]" keeping the original sentence-ending
        # punctuation.
        return re.sub(r"([\.:?!])\f", r"\1 [NEW_PAGE]", text)
