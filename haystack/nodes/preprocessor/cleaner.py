from typing import List, Optional, Iterable

import logging
import re
from copy import deepcopy
import warnings
from math import inf

from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes.preprocessor.splitter import split_by_regex


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

        self.clean_whitespace = clean_whitespace
        self.clean_header_footer = clean_header_footer
        self.clean_empty_lines = clean_empty_lines
        self.clean_regex = clean_regex
        self.header_footer_n_chars = header_footer_n_chars
        self.header_footer_pages_to_ignore = header_footer_pages_to_ignore

    def run(  # type: ignore
        self,
        documents: List[Document],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: List[int] = None,
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

        if not isinstance(header_footer_n_chars, int) or header_footer_n_chars < 0:
            raise ValueError("header_footer_n_chars must be an integer >= 0")

        if any(not isinstance(page, int) or page < 0 for page in header_footer_pages_to_ignore):
            raise ValueError("header_footer_pages_to_ignore must contain only integers >= 0")

        clean_docs = [
            self.clean(
                document=document,
                clean_whitespace=clean_whitespace,
                clean_header_footer=clean_header_footer,
                clean_empty_lines=clean_empty_lines,
                clean_regex=clean_regex,
                header_footer_n_chars=header_footer_n_chars,
                header_footer_pages_to_ignore=header_footer_pages_to_ignore,
            )
            for document in documents
        ]
        return {"documents": clean_docs}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: List[int] = None,
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

    def clean(
        self,
        document: Document,
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: List[int] = [],
    ) -> Document:
        """
        Perform document cleaning on a single document and return a single document.
        This method will deal with whitespaces, headers, footers and empty lines.

        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                        for the longest common string. This heuristic uses exact matches and therefore
                                        works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                        or similar.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param clean_regex: Remove the specified regex matches from the text. For example, `clean_regex='[0-9]'`
                            removes all digits from the document's content, and `clean_regex='(a string|another string)'`
                            will remove all occurrences of either string from the document content.
        :param header_footer_n_chars: how many chars to look for headers and footer in.
        :param header_footer_pages_to_ignore: which pages to ignore in the header-footer detection heuristic
        """
        if isinstance(document, dict):
            warnings.warn(
                "Passing a dictionary to Preprocessor.clean() is deprecated. Use Document objects.", DeprecationWarning
            )
            document = Document.from_dict(document)

        if document.content_type != "text":
            raise ValueError(
                f"Document content type is not 'text', but '{document.content_type}'. Preprocessor only handles text documents."
            )

        clean_document = deepcopy(document)

        if clean_header_footer:
            clean_document = remove_header_footer(
                document=clean_document, n_chars=header_footer_n_chars, pages_to_ignore=header_footer_pages_to_ignore
            )

        if clean_whitespace:
            # Whitespace around page breaks
            clean_document = replace_regex_matches(
                document=clean_document, pattern=r"[ \t\r\v]*\f[ \t\r\v]*", string="\f"
            )
            # Whitespace around newlines
            clean_document = replace_regex_matches(
                document=clean_document, pattern=r"[ \t\r\v]*\n[ \t\r\v]*", string="\n"
            )
            # Leading and trailing spaces
            clean_document = replace_regex_matches(document=clean_document, pattern=r"^[ \t\r\v]*", string="")
            clean_document = replace_regex_matches(document=clean_document, pattern=r"[ \t\r\v]*$", string="")

        if clean_empty_lines:
            clean_document = replace_regex_matches(document=clean_document, pattern=r"[\n]{2,}", string="\n")

        if clean_regex:
            clean_document = replace_regex_matches(document=clean_document, pattern=clean_regex, string="")

        return clean_document


def remove_header_footer(
    document: Document, n_chars: int = 100, pages_to_ignore: List[int] = [], min_len: int = 5, max_len: int = 50
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
        document = replace_regex_matches(document, pattern=rf"{escaped_header}", string="")
        logger.debug("Removed header: %s from doc id %s", header, document.id)

    footer = longest_common_suffix(texts=[page[-n_chars:] for page in pages], min_len=min_len, max_len=max_len)
    if footer:
        escaped_footer = "".join([rf"\{char}" if char in REGEX_METACHARS else char for char in footer])
        document = replace_regex_matches(document, pattern=rf"{escaped_footer}", string="")
        logger.debug("Removed footer: %s from doc id %s", footer, document.id)

    return document


def replace_regex_matches(document: Document, pattern: str, string: str) -> Document:
    """
    Replaces every match of the given regex in the text with the given string and
    re-aligns the headlines positions if they were present in the meta.

    :param document: the document to clean of whitespace
    :param substrings: the substrings to remove from the text
    :return: the document cleaned of whitespace, with the headlines positions re-aligned
                if headlines were present in the meta.
    """
    if not pattern or pattern == "()":
        return document

    headlines = deepcopy(document.meta.get("headlines", None)) or []

    # Clean the documents by splitting them on the clean regex and removing the match
    units, offsets = split_by_regex(text=document.content, pattern=pattern, _clean_separator=True)
    new_content = string.join(units)

    # check for a trailing match that might have been left out in the above cleanup
    trailing_match = re.compile(rf"{pattern}$").search(document.content)
    if trailing_match:
        new_content += string

    document.content = new_content

    # Shift all headlines by the introduced offset
    position_in_document = 0
    for unit, offset in zip(units, offsets):
        for headline in headlines:
            if headline["start_idx"] > position_in_document:
                headline["start_idx"] -= offset
        position_in_document += len(unit) + offset

    # Resize headlines by replacing the matches and shifting their boundaries if necessary
    new_headlines = []
    compiled_pattern = re.compile(pattern)
    for headline in headlines:

        # Check for a match at the start that would force us to shift start_idx rightward
        match_start = compiled_pattern.match(headline["content"])
        if match_start:
            headline["start_idx"] += match_start.end() - match_start.start()

        # If the headline contains the pattern to remove, take it out
        headline["content"] = compiled_pattern.sub(string, headline["content"])
        # Some headlines might get fully erased at this stage
        if headline["content"]:
            new_headlines.append(headline)

    if new_headlines:
        document.meta["headlines"] = new_headlines

    return document


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
