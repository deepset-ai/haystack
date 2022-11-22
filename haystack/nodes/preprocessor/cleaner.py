from typing import List, Optional, Iterable

import logging
import re
from copy import deepcopy
import warnings
from math import inf

from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes.preprocessor.splitter import split_by_regex, DocumentSplitter
from haystack.nodes.preprocessor.merger import DocumentMerger


logger = logging.getLogger(__name__)


REGEX_METACHARS = r".^$*+?{}[]\|()"


class DocumentCleaner(BaseComponent):

    outgoing_edges = 1

    def __init__(
        self,
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: List[int] = None,
    ):
        super().__init__()

        if not header_footer_pages_to_ignore:
            header_footer_pages_to_ignore = []
        self._validate_clean_parameters(
            header_footer_n_chars=header_footer_n_chars, header_footer_pages_to_ignore=header_footer_pages_to_ignore
        )

        self.clean_whitespace = clean_whitespace
        self.clean_header_footer = clean_header_footer
        self.clean_empty_lines = clean_empty_lines
        self.clean_regex = clean_regex
        self.header_footer_n_chars = header_footer_n_chars
        self.header_footer_pages_to_ignore = header_footer_pages_to_ignore

        self.splitter = DocumentSplitter(split_by="regex", split_length=1)
        self.merger = DocumentMerger(window_size=0, realign_headlines=True, retain_page_number=True)

    def _validate_clean_parameters(self, header_footer_n_chars, header_footer_pages_to_ignore):
        """
        Performs some validation on the input parameters.
        """
        if not isinstance(header_footer_n_chars, int) or header_footer_n_chars < 0:
            raise ValueError("header_footer_n_chars must be an integer >= 0")

        if not isinstance(header_footer_pages_to_ignore, Iterable):
            raise ValueError(
                "header_footer_pages_to_ignore must be an iterable of integers, "
                "referring to the numbers of the pages to ignore: for example [1, 2, 3, -1, -2, -3] "
                "will ignore the first and last three pages."
            )

        if any(not isinstance(page, int) for page in header_footer_pages_to_ignore):
            raise ValueError("header_footer_pages_to_ignore must contain only integers")

    def run(  # type: ignore
        self,
        documents: List[Document],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
    ):
        clean_whitespace = clean_whitespace if clean_whitespace is not None else self.clean_whitespace
        clean_header_footer = clean_header_footer if clean_header_footer is not None else self.clean_header_footer
        clean_empty_lines = clean_empty_lines if clean_empty_lines is not None else self.clean_empty_lines
        clean_regex = clean_regex if clean_regex is not None else self.clean_regex
        header_footer_n_chars = (
            header_footer_n_chars if header_footer_n_chars is not None else self.header_footer_n_chars
        )
        header_footer_pages_to_ignore = (
            header_footer_pages_to_ignore
            if header_footer_pages_to_ignore is not None
            else self.header_footer_pages_to_ignore
        )
        self._validate_clean_parameters(
            header_footer_n_chars=header_footer_n_chars, header_footer_pages_to_ignore=header_footer_pages_to_ignore
        )

        # Fail early
        if any(document.content_type != "text" for document in documents):
            raise ValueError(
                f"Document content type is not 'text', but '{document.content_type}'. Preprocessor only handles text documents."
            )

        clean_docs = []
        for document in documents:
            if isinstance(document, dict):
                warnings.warn(
                    "Passing a dictionary to Preprocessor.clean() is deprecated. Use Document objects.",
                    DeprecationWarning,
                )
                document = Document.from_dict(document)
            document = deepcopy(document)

            if clean_header_footer:
                document = self.remove_header_footer(
                    document=document, n_chars=header_footer_n_chars, pages_to_ignore=header_footer_pages_to_ignore
                )

            if clean_whitespace:
                # Whitespace around page breaks
                document = self.replace_regex_matches(
                    document=document, pattern=r"[ \t\r\v]*\f[ \t\r\v]*", replacement="\f"
                )
                # Whitespace around newlines
                document = self.replace_regex_matches(
                    document=document, pattern=r"[ \t\r\v]*\n[ \t\r\v]*", replacement="\n"
                )
                # Leading spaces
                document = self.replace_regex_matches(document=document, pattern=r"^[ \t\r\v]*", replacement="")
                # Trailing spaces
                document = self.replace_regex_matches(document=document, pattern=r"[ \t\r\v]*$", replacement="")

            if clean_empty_lines:
                document = self.replace_regex_matches(document=document, pattern=r"[\n]{2,}", replacement="\n")

            if clean_regex:
                document = self.replace_regex_matches(document=document, pattern=clean_regex, replacement="")

            clean_docs.append(document)
        return {"documents": clean_docs}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
    ):
        documents = [
            self.run(
                document=document,
                clean_whitespace=clean_whitespace,
                clean_header_footer=clean_header_footer,
                clean_empty_lines=clean_empty_lines,
                clean_regex=clean_regex,
                header_footer_n_chars=header_footer_n_chars,
                header_footer_pages_to_ignore=header_footer_pages_to_ignore,
            )[0]["documents"]
            for document in documents
        ]
        return {"documents": documents}, "output_1"

    def remove_header_footer(
        self,
        document: Document,
        n_chars: int = 100,
        pages_to_ignore: List[int] = [],
        min_len: int = 5,
        max_len: int = 50,
    ) -> Document:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common prefix/suffix.
        For headers we only search in the first n_chars characters, for footers we search in the last n_chars.

        Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
        but won't detect "Page 3 of 4" or similar. For those, use `clean_regex`.

        :param document: the document to remove headers and footers from.
        :param n_chars: number of first/last characters where the header/footer shall be searched in
        :param pages_to_ignore: numbers of the pages to ignore (e.g. TOCs often don't contain footer/header)
        :param min_len: how many chars, minimum, the header/footer can be made of
        :param max_len: how many chars, maximum, the header/footer can be made of
        """
        pages = document.content.split("\f")
        pages = [page for page_number, page in enumerate(pages) if page_number not in pages_to_ignore]

        # empty pages are a typical issue for header/footer detection.
        # Clean them up separately to avoid messing up the page numbering
        pages = [page for page in pages if page.strip()]

        header = longest_common_prefix(texts=[page[:n_chars] for page in pages], min_len=min_len, max_len=max_len)
        if header:
            escaped_header = "".join([rf"\{char}" if char in REGEX_METACHARS else char for char in header])
            document = self.replace_regex_matches(document, pattern=rf"{escaped_header}", replacement="")
            logger.debug("Removed header: %s from doc id %s", header, document.id)

        footer = longest_common_suffix(texts=[page[-n_chars:] for page in pages], min_len=min_len, max_len=max_len)
        if footer:
            escaped_footer = "".join([rf"\{char}" if char in REGEX_METACHARS else char for char in footer])
            document = self.replace_regex_matches(document, pattern=rf"{escaped_footer}", replacement="")
            logger.debug("Removed footer: %s from doc id %s", footer, document.id)

        return document

    def replace_regex_matches(self, document: Document, pattern: str, replacement: str) -> Document:
        """
        Replaces every match of the given regex in the text with the given string and
        re-aligns the headlines positions if they were present in the meta.

        :param document: the document to clean of whitespace
        :param pattern: the pattern to match for the substrings to remove from the text

        :return: the document cleaned of whitespace, with the headlines positions re-aligned
                    if headlines were present in the meta.
        """
        if not pattern or pattern == "()":
            return document

        # Split the docs on the regex to clean, so that the part to remove will always be at the tail
        units = self.splitter.split_into_units(
            document=document, units=split_by_regex(text=document.content, pattern=pattern), add_page_number=False
        )
        # Remove the offsets and re-check the headlines
        for doc, offset in zip(*units):

            # Remove matches from the headlines contents and take out empty ones
            remaining_headlines = []
            compiled_pattern = re.compile(pattern)
            if "headlines" in doc.meta.keys() and doc.meta["headlines"] is not None:
                for headline in doc.meta["headlines"]:
                    # If the headline contains the pattern to remove somewhere else, take it out
                    headline["content"] = compiled_pattern.sub(replacement, headline["content"])
                    # Some headlines might get fully erased at this stage
                    if headline["content"]:
                        remaining_headlines.append(headline)

                doc.meta["headlines"] = remaining_headlines

            if offset:
                # Find headlines that were contained in a match
                remaining_headlines = []
                if "headlines" in doc.meta.keys() and doc.meta["headlines"] is not None:
                    for headline in doc.meta["headlines"]:
                        if not (
                            len(doc.content) - offset <= headline["start_idx"]
                            and headline["content"] in doc.content[-offset:]
                        ):
                            remaining_headlines.append(headline)
                    doc.meta["headlines"] = remaining_headlines

                # Remove the match from the document content too
                doc.content = doc.content[:-offset]

        # Merge the documents back
        clean_document = self.merger.run(
            documents=units[0], separator=replacement, window_size=0, realign_headlines=True, retain_page_number=True
        )[0]["documents"][0]

        # check for a trailing match that might have been added in the above cleanup
        trailing_match = re.compile(rf"{pattern}$").search(document.content)
        if trailing_match:
            clean_document.content += replacement

        return clean_document


def longest_common_prefix(texts: list[str], min_len: int, max_len: int) -> Optional[str]:
    """
    Find the longest common prefix across several texts. used for header detection.

    :param texts: list of strings that shall be searched for common prefix
    :param min_len: maximum length to consider
    :param max_len: minimum length to consider
    :return: longest common prefix in all given texts
    """
    if not min_len > 0 or not max_len > 0:
        raise ValueError("Prefix length must be > 0")

    texts = sorted(texts, key=lambda x: len(x))
    for prefix_len in reversed(range(min_len, min(len(texts[0]), max_len) + 1)):
        prefix = texts[0][:prefix_len]
        if all(text[:prefix_len] == prefix for text in texts):
            return prefix
    return None


def longest_common_suffix(texts: list[str], min_len: int, max_len: int) -> Optional[str]:
    """
    Find the longest common suffix across several texts. used for footer detection.

    :param texts: list of strings that shall be searched for common suffix
    :param min_len: maximum length to consider
    :param max_len: minimum length to consider
    :return: longest common suffix in all given texts
    """
    if not min_len > 0 or not max_len > 0:
        raise ValueError("Suffix length must be > 0")

    texts = sorted(texts, key=lambda x: len(x))
    for suffix_len in reversed(range(min_len, min(len(texts[0]), max_len) + 1)):
        suffix = texts[0][len(texts[0]) - suffix_len :]
        if all(text[len(text) - suffix_len :] == suffix for text in texts):
            return suffix
    return None
