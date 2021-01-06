import logging
import re
from copy import deepcopy
from functools import partial, reduce
from itertools import chain
from typing import List, Optional, Generator, Set

import nltk
from more_itertools import windowed

from haystack.preprocessor.base import BasePreProcessor

logger = logging.getLogger(__name__)


class PreProcessor(BasePreProcessor):
    def __init__(
        self,
        clean_whitespace: Optional[bool] = True,
        clean_header_footer: Optional[bool] = False,
        clean_empty_lines: Optional[bool] = True,
        split_by: Optional[str] = "word",
        split_length: Optional[int] = 1000,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = True,
    ):
        """
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param split_by: Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by ->
                           "sentence", then each output document will have 10 sentences.
        :param split_overlap: Word overlap between two adjacent documents after a split.
                              Setting this to a positive number essentially enables the sliding window approach.
                              For example, if split_by -> `word`,
                              split_length -> 5 & split_overlap -> 2, then the splits would be like:
                              [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12].
                              Set the value to None to ensure there is no overlap among the documents after splitting.
        :param split_respect_sentence_boundary: Whether to split in partial sentences if split_by -> `word`. If set
                                                to True, the individual split will always have complete sentences &
                                                the number of words will be <= split_length.
        """
        nltk.download("punkt")
        self.clean_whitespace = clean_whitespace
        self.clean_header_footer = clean_header_footer
        self.clean_empty_lines = clean_empty_lines
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_respect_sentence_boundary = split_respect_sentence_boundary

    def clean(self, document: dict) -> dict:
        """
        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().
        """
        text = document["text"]
        if self.clean_header_footer:
            text = self._find_and_remove_header_footer(
                text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )

        if self.clean_whitespace:
            lines = text.splitlines()

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)

        if self.clean_empty_lines:
            text = re.sub(r"\n\n+", "\n\n", text)

        document["text"] = text
        return document

    def split(self, document: dict) -> List[dict]:
        """Perform document splitting on a single document. This method can split on different units, at different lengths,
        with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
        the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents. """

        if not self.split_by:
            return [document]

        if not self.split_length:
            raise Exception("split_length needs be set when using split_by.")

        if self.split_respect_sentence_boundary and self.split_by not in("word","sentence"):
            raise NotImplementedError("'split_respect_sentence_boundary=True' is only compatible with"
                                      " split_by='word' or split_by='sentence'.")

        text = document["text"]

        if self.split_respect_sentence_boundary and self.split_by == "word":
            # split by words ensuring no sub sentence splits
            sentences = nltk.tokenize.sent_tokenize(text)
            word_count = 0
            list_splits = []
            current_slice: List[str] = []
            for sen in sentences:
                current_word_count = len(sen.split(" "))
                if current_word_count > self.split_length:
                    logger.warning(f"A sentence found with word count higher than the split length.")
                if word_count + current_word_count > self.split_length:
                    list_splits.append(current_slice)
                    #Enable split_stride with split_by='word' while respecting sentence boundaries.
                    if self.split_overlap:
                        overlap = []
                        w_count = 0
                        for s in current_slice[::-1]:
                            sen_len = len(s.split(" "))
                            if w_count < self.split_overlap:
                                overlap.append(s)
                                w_count += sen_len
                            else:
                                break
                        current_slice = list(reversed(overlap))
                        word_count = w_count
                    else:
                        current_slice = []
                        word_count = 0
                current_slice.append(sen)
                word_count += len(sen.split(" "))
            if current_slice:
                list_splits.append(current_slice)
            text_splits = [' '.join(sl) for sl in list_splits]
        else:
            # create individual "elements" of passage, sentence, or word
            if self.split_by == "passage":
                elements = text.split("\n\n")
            elif self.split_by == "sentence":
                elements = nltk.tokenize.sent_tokenize(text)
            elif self.split_by == "word":
                elements = text.split(" ")
            else:
                raise NotImplementedError("PreProcessor only supports 'passage' or 'sentence' split_by options.")

            # concatenate individual elements based on split_length & split_stride
            if self.split_overlap:
                segments = windowed(elements, n=self.split_length, step=self.split_length - self.split_overlap)
            else:
                segments = windowed(elements, n=self.split_length, step=self.split_length)
            text_splits = []
            for seg in segments:
                txt = " ".join([t for t in seg if t])
                text_splits.append(txt)

        # create new document dicts for each text split
        documents = []
        for i, txt in enumerate(text_splits):
            doc = deepcopy(document)
            doc["text"] = txt
            if "meta" not in doc.keys() or doc["meta"] is None:
                doc["meta"] = {}
            doc["meta"]["_split_id"] = i
            documents.append(doc)

        return documents

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
        logger.debug(f"Removed header '{found_header}' and footer '{found_footer}' in document")
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
