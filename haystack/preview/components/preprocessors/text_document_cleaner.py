import re
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import Any, Dict, List, Optional, Generator, Set

from haystack.preview import Document, component, default_from_dict, default_to_dict


@component
class TextDocumentCleaner:
    """
    Makes text documents more readable by cleaning empty lines, extra whitespaces, headers and footers, etc.
    This is useful for preparing the documents for further processing by LLMs.
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
        :param remove_repeated_substrings: Whether to remove repeated substrings, such as headers and footers.
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
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("TextDocumentCleaner expects a List of Documents as input.")

        cleaned_docs = []
        for doc in documents:
            if doc.text is None:
                raise ValueError(
                    f"TextDocumentCleaner only works with text documents but document.text for document ID {doc.id} is None."
                )
            text = doc.text

            if self.remove_empty_lines:
                text = self._remove_empty_lines(text)
            if self.remove_extra_whitespaces:
                text = self._remove_extra_whitespaces(text)
            if self.remove_repeated_substrings:
                text = self._remove_repeated_substrings(text)
            if self.remove_substrings:
                text = self._remove_substrings(text, self.remove_substrings)
            if self.remove_regex:
                text = self._remove_regex(text, self.remove_regex)

            cleaned_docs.append(Document(text=text, metadata=deepcopy(doc.metadata)))

        return {"documents": cleaned_docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            clean_empty_lines=self.remove_empty_lines,
            clean_whitespaces=self.remove_extra_whitespaces,
            clean_repeated_substrings=self.remove_repeated_substrings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextDocumentCleaner":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove empty lines and lines that contain nothing but whitespaces from text.
        :param text: Text to clean.
        """
        lines = text.split("\n")
        non_empty_lines = filter(lambda line: line.strip() != "", lines)
        return "\n".join(non_empty_lines)

    def _remove_extra_whitespaces(self, text: str) -> str:
        """
        Remove extra whitespaces from text.
        :param text: Text to clean.
        """
        return re.sub(r"\s\s+", " ", text).strip()

    def _remove_regex(self, text: str, regex: str) -> str:
        """
        Remove substrings that match the specified regex from the text.
        :param text: Text to clean.
        :param regex: Regex to match and replace substrings by "".
        """
        return re.sub(regex, "", text).strip()

    def _remove_substrings(self, text: str, substrings: List[str]) -> str:
        """
        Remove all specified substrings from the text.
        :param text: Text to clean.
        :param substrings: Substrings to remove.
        """
        for substring in substrings:
            text = text.replace(substring, "")
        return text

    def _remove_repeated_substrings(self, text: str) -> str:
        return self._find_and_remove_header_footer(
            text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
        )

    def _find_and_remove_header_footer(
        self, text: str, n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
    ) -> str:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common string.
        For headers, we only search in the first n_chars characters (for footer: last n_chars).
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
        # logger.debug("Removed header '%s' and footer '%s' in document", found_header, found_footer)
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
