from abc import abstractmethod
from functools import partial, reduce
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple, Generator, Set, Dict, Any

import langdetect


class BaseConverter:
    """
    Base class for implementing file converts to transform input documents to text format for ingestion in DocumentStore.
    """

    def __init__(
        self,
        remove_numeric_tables: Optional[bool] = None,
        remove_header_footer: Optional[bool] = None,
        remove_whitespace: Optional[bool] = None,
        remove_empty_lines: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param remove_header_footer: use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param remove_whitespace: strip whitespaces before or after each line in the text.
        :param remove_empty_lines: remove more than two empty lines in the text.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        self.remove_numeric_tables = remove_numeric_tables
        self.remove_header_footer = remove_header_footer
        self.remove_whitespace = remove_whitespace
        self.remove_empty_lines = remove_empty_lines
        self.valid_languages = valid_languages

    @abstractmethod
    def convert(self, file_path: Path, meta: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """
        Convert a file to a dictionary containing the text and any associated meta data.

        File converters may extract file meta like name or size. In addition to it, user
        supplied meta data like author, url, external IDs can be supplied as a dictionary.

        :param file_path: path of the file to convert
        :param meta: dictionary of meta data key-value pairs to append in the returned document.
        """
        pass

    def validate_language(self, text: str) -> bool:
        """
        Validate if the language of the text is one of valid languages.
        """
        if not self.valid_languages:
            return True

        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang in self.valid_languages:
            return True
        else:
            return False

    def find_and_remove_header_footer(
            self, pages: List[str], n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
    ) -> Tuple[List[str], Optional[str], Optional[str]]:
        """
        Heuristic to find footers and headers across different pages by searching for the longest common string.
        For headers we only search in the first n_chars characters (for footer: last n_chars).
        Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
         but won't detect "Page 3 of 4" or similar.

        :param pages: list of strings, one string per page
        :param n_chars: number of first/last characters where the header/footer shall be searched in
        :param n_first_pages_to_ignore: number of first pages to ignore (e.g. TOCs often don't contain footer/header)
        :param n_last_pages_to_ignore: number of last pages to ignore
        :return: (cleaned pages, found_header_str, found_footer_str)
        """

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
        return pages, found_header, found_footer

    def _ngram(self, seq: str, n: int) -> Generator[str, None, None]:
        """
        Return ngram (of tokens - currently splitted by whitespace)
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
            " ".join(words[i: i + n]).replace(" \n", "\n").replace(" \t", "\t") for i in range(0, len(words) - n + 1)
        )

        return ngrams

    def _allngram(self, seq: str, min_ngram: int, max_ngram: int) -> Set[str]:
        lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
        ngrams = map(partial(self._ngram, seq), lengths)
        res = set(chain.from_iterable(ngrams))
        return res

    def _find_longest_common_ngram(self, sequences: List[str], max_ngram: int = 30, min_ngram: int = 3) -> Optional[str]:
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

