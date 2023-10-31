import logging
import re
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import Generator, List, Optional, Set

from haystack.preview import Document, component

logger = logging.getLogger(__name__)


@component
class DocumentCleaner:
    """
    Makes text documents more readable by removing extra whitespaces, empty lines, specified substrings, regexes, page headers and footers (in this order).
    This is useful for preparing the documents for further processing by LLMs.

    Example usage in an indexing pipeline:

    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TextFileToDocument(), name="text_file_converter")
    p.add_component(instance=DocumentCleaner(), name="cleaner")
    p.add_component(instance=TextDocumentSplitter(split_by="sentence", split_length=1), name="splitter")
    p.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    p.connect("text_file_converter.documents", "cleaner.documents")
    p.connect("cleaner.documents", "splitter.documents")
    p.connect("splitter.documents", "writer.documents")
    ```
    """

    def __init__(
        self,
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
    ):
        """
        :param remove_empty_lines: Whether to remove empty lines.
        :param remove_extra_whitespaces: Whether to remove extra whitespaces.
        :param remove_repeated_substrings: Whether to remove repeated substrings (headers/footers) from pages.
            Pages in the text need to be separated by form feed character "\f",
            which is supported by TextFileToDocument and AzureOCRDocumentConverter.
        :param remove_substrings: List of substrings to remove from the text.
        :param remove_regex: Regex to match and replace substrings by "".
        """

        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_repeated_substrings = remove_repeated_substrings
        self.remove_substrings = remove_substrings
        self.remove_regex = remove_regex

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Run the DocumentCleaner on the given list of documents.
        The documents' metadata remain unchanged.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("DocumentCleaner expects a List of Documents as input.")

        cleaned_docs = []
        for doc in documents:
            if doc.content is None:
                logger.warning(
                    "DocumentCleaner only cleans text documents but document.content for document ID %s is None.",
                    doc.id,
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

            cleaned_docs.append(Document(content=text, meta=deepcopy(doc.meta)))

        return {"documents": cleaned_docs}

    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines and lines that contain nothing but whitespaces from text.
        :param text: Text to clean.
        :param return: The text without empty lines.
        """
        lines = text.split("\n")
        non_empty_lines = filter(lambda line: line.strip() != "", lines)
        return "\n".join(non_empty_lines)

    def _remove_extra_whitespaces(self, text: str) -> str:
        """
        Remove extra whitespaces from text.
        :param text: Text to clean.
        :param return: The text without extra whitespaces.
        """
        return re.sub(r"\s\s+", " ", text).strip()

    def _remove_regex(self, text: str, regex: str) -> str:
        """
        Remove substrings that match the specified regex from the text.
        :param text: Text to clean.
        :param regex: Regex to match and replace substrings by "".
        :param return: The text without any substrings that match the regex.
        """
        return re.sub(regex, "", text).strip()

    def _remove_substrings(self, text: str, substrings: List[str]) -> str:
        """
        Remove all specified substrings from the text.
        :param text: Text to clean.
        :param substrings: Substrings to remove.
        :return: The text without the specified substrings.
        """
        for substring in substrings:
            text = text.replace(substring, "")
        return text

    def _remove_repeated_substrings(self, text: str) -> str:
        """
        Remove any substrings from the text that occur repeatedly on every page. For example headers or footers.
        Pages in the text need to be separated by form feed character "\f".
        :param text: Text to clean.
        :return: The text without the repeated substrings.
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
        :return: The text without the found headers and footers.
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
        Return all ngrams of length n from a text sequence. Each ngram consists of n words split by whitespace.
        :param seq: The sequence to generate ngrams from.
        :param n: The length of the ngrams to generate.
        :return: A Generator generating all ngrams of length n from the given sequence.
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
        :return: A set of all ngrams from the given sequence.
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
        :return: The longest ngram that all sequences have in common.
        """
        sequences = [s for s in sequences if s]  # filter empty sequences
        if not sequences:
            return ""
        seqs_ngrams = map(partial(self._allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)

        longest = max(intersection, key=len, default="")
        return longest if longest.strip() else ""
