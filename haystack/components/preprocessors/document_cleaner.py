# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import Any, Callable, Dict, Generator, List, Optional, Set

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)


def DEFAULT_ID_GENERATOR(old_doc: Document, new_doc: Document) -> str:
    """
    Default id_generator for the DocumentCleaner.
    """
    return new_doc.id


def KEEP_ID(old_doc: Document, new_doc: Document) -> str:
    """
    Sample id_generator to keep the original ids for the cleaned documents.
    """
    return old_doc.id


@component
class DocumentCleaner:
    """
    Cleans the text in the documents.

    Cleans up text documents by removing extra whitespaces, empty lines, specified substrings, regexes,
    page headers and footers (in this order).

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.preprocessors import DocumentCleaner

    doc = Document(content="This   is  a  document  to  clean\\n\\n\\nsubstring to remove")

    cleaner = DocumentCleaner(remove_substrings = ["substring to remove"])
    result = cleaner.run(documents=[doc])

    assert result["documents"][0].content == "This is a document to clean "
    ```
    """

    def __init__(
        self,
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
        id_generator: Callable[[Document, Document], str] = DEFAULT_ID_GENERATOR,
    ):
        """
        Initialize the DocumentCleaner.

        :param remove_empty_lines: Whether to remove empty lines.
        :param remove_extra_whitespaces: Whether to remove extra whitespaces.
        :param remove_repeated_substrings: Whether to remove repeated substrings (headers/footers) from pages.
            Pages in the text need to be separated by form feed character "\\f",
            which is supported by `TextFileToDocument` and `AzureOCRDocumentConverter`.
        :param remove_substrings: List of substrings to remove from the text.
        :param remove_regex: Regex to match and replace substrings by "".
        :param id_generator: Function that expects the original and new documents and returns the id for the new document.
        """

        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_repeated_substrings = remove_repeated_substrings
        self.remove_substrings = remove_substrings
        self.remove_regex = remove_regex
        self.id_generator = id_generator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        serialized_id_generator = serialize_callable(self.id_generator)
        return default_to_dict(
            self,
            remove_empty_lines=self.remove_empty_lines,
            remove_extra_whitespaces=self.remove_extra_whitespaces,
            remove_repeated_substrings=self.remove_repeated_substrings,
            remove_substrings=self.remove_substrings,
            remove_regex=self.remove_regex,
            id_generator=serialized_id_generator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentCleaner":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_id_generator = init_params.get("id_generator")
        if serialized_id_generator:
            data["init_parameters"]["id_generator"] = deserialize_callable(serialized_id_generator)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Cleans up the documents.

        :param documents: List of Documents to clean.

        :returns: A dictionary with the following key:
            - `documents`: List of cleaned Documents.

        :raises TypeError: if documents is not a list of Documents.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("DocumentCleaner expects a List of Documents as input.")

        cleaned_docs = []
        for doc in documents:
            if doc.content is None:
                logger.warning(
                    "DocumentCleaner only cleans text documents but document.content for document ID %{document_id} is None.",
                    document_id=doc.id,
                )
                cleaned_docs.append(doc)
                continue
            text = doc.content

            if self.remove_extra_whitespaces:
                text = self._remove_extra_whitespaces(text)
            if self.remove_empty_lines:
                text = self._remove_empty_lines(text)
            if self.remove_substrings:
                text = self._remove_substrings(text, self.remove_substrings)
            if self.remove_regex:
                text = self._remove_regex(text, self.remove_regex)
            if self.remove_repeated_substrings:
                text = self._remove_repeated_substrings(text)

            cleaned_doc = Document(content=text, meta=deepcopy(doc.meta))
            cleaned_doc.id = self.id_generator(doc, cleaned_doc)
            cleaned_docs.append(cleaned_doc)

        return {"documents": cleaned_docs}

    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines and lines that contain nothing but whitespaces from text.

        :param text: Text to clean.
        :returns: The text without empty lines.
        """
        lines = text.split("\n")
        non_empty_lines = filter(lambda line: line.strip() != "", lines)
        return "\n".join(non_empty_lines)

    def _remove_extra_whitespaces(self, text: str) -> str:
        """
        Remove extra whitespaces from text.

        :param text: Text to clean.
        :returns: The text without extra whitespaces.
        """
        return re.sub(r"\s\s+", " ", text).strip()

    def _remove_regex(self, text: str, regex: str) -> str:
        """
        Remove substrings that match the specified regex from the text.

        :param text: Text to clean.
        :param regex: Regex to match and replace substrings by "".
        :returns: The text without the substrings that match the regex.
        """
        return re.sub(regex, "", text).strip()

    def _remove_substrings(self, text: str, substrings: List[str]) -> str:
        """
        Remove all specified substrings from the text.

        :param text: Text to clean.
        :param substrings: Substrings to remove.
        :returns: The text without the specified substrings.
        """
        for substring in substrings:
            text = text.replace(substring, "")
        return text

    def _remove_repeated_substrings(self, text: str) -> str:
        """
        Remove any substrings from the text that occur repeatedly on every page. For example headers or footers.

        Pages in the text need to be separated by form feed character "\f".
        :param text: Text to clean.
        :returns: The text without the repeated substrings.
        """
        return self._find_and_remove_header_footer(
            text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
        )

    def _find_and_remove_header_footer(
        self, text: str, n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
    ) -> str:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common string.

        Pages in the text need to be separated by form feed character "\f".
        For headers, we only search in the first n_chars characters (for footer: last n_chars).
        Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
         but won't detect "Page 3 of 4" or similar.

        :param n_chars: The number of first/last characters where the header/footer shall be searched in.
        :param n_first_pages_to_ignore: The number of first pages to ignore (e.g. TOCs often don't contain footer/header).
        :param n_last_pages_to_ignore: The number of last pages to ignore.
        :returns: The text without the found headers and footers.
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

        logger.debug(
            "Removed header '{header}' and footer '{footer}' in document", header=found_header, footer=found_footer
        )
        text = "\f".join(pages)
        return text

    def _ngram(self, seq: str, n: int) -> Generator[str, None, None]:
        """
        Return all ngrams of length n from a text sequence. Each ngram consists of n words split by whitespace.

        :param seq: The sequence to generate ngrams from.
        :param n: The length of the ngrams to generate.
        :returns: A Generator generating all ngrams of length n from the given sequence.
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
        """
        Generates all possible ngrams from a given sequence of text.

        Considering all ngram lengths between the minimum and maximum length.

        :param seq: The sequence to generate ngrams from.
        :param min_ngram: The minimum length of ngram to consider.
        :param max_ngram: The maximum length of ngram to consider.
        :returns: A set of all ngrams from the given sequence.
        """
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        res = set(chain.from_iterable(ngrams))
        return res

    def _find_longest_common_ngram(self, sequences: List[str], min_ngram: int = 3, max_ngram: int = 30) -> str:
        """
        Find the longest common ngram across a list of text sequences (e.g. start of pages).

        Considering all ngram lengths between the minimum and maximum length. Helpful for finding footers, headers etc.
        Empty sequences are ignored.

        :param sequences: The list of strings that shall be searched for common n_grams.
        :param max_ngram: The maximum length of ngram to consider.
        :param min_ngram: The minimum length of ngram to consider.
        :returns: The longest ngram that all sequences have in common.
        """
        sequences = [s for s in sequences if s]  # filter empty sequences
        if not sequences:
            return ""
        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)

        longest = max(intersection, key=len, default="")
        return longest if longest.strip() else ""
