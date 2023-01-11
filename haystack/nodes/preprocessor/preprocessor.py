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
        clean_empty_lines: bool = True,
        remove_substrings: List[str] = [],
        split_by: Optional[Literal["word", "sentence", "passage"]] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_respect_sentence_boundary: bool = True,
        tokenizer_model_folder: Optional[Union[str, Path]] = None,
        language: str = "en",
        id_hash_keys: Optional[List[str]] = None,
        progress_bar: bool = True,
        add_page_number: bool = False,
    ):
        """
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                     or similar.
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
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
                                `AzureConverter`.
        """
        super().__init__()

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt")
            except FileExistsError as error:
                logger.debug(f"NLTK punkt tokenizer seems to be already downloaded. Error message: {error}")
                pass
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
        self.print_log: Set[str] = set()
        self.id_hash_keys = id_hash_keys
        self.progress_bar = progress_bar
        self.add_page_number = add_page_number

    def process(
        self,
        documents: Union[dict, Document, List[Union[dict, Document]]],
        clean_whitespace: Optional[bool] = None,
        clean_header_footer: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        remove_substrings: List[str] = [],
        split_by: Optional[Literal["word", "sentence", "passage"]] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:

        """
        Perform document cleaning and splitting. Can take a single document or a list of documents as input and returns a list of documents.
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
        split_by: Optional[Literal["word", "sentence", "passage"]] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:

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

        cleaned_document = self.clean(
            document=document,
            clean_whitespace=clean_whitespace,
            clean_header_footer=clean_header_footer,
            clean_empty_lines=clean_empty_lines,
            remove_substrings=remove_substrings,
            id_hash_keys=id_hash_keys,
        )
        split_documents = self.split(
            document=cleaned_document,
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_respect_sentence_boundary=split_respect_sentence_boundary,
            id_hash_keys=id_hash_keys,
        )
        return split_documents

    def _process_batch(
        self, documents: List[Union[dict, Document]], id_hash_keys: Optional[List[str]] = None, **kwargs
    ) -> List[Document]:
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
    ) -> Document:
        """
        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines. Its exact functionality is defined by the parameters passed into PreProcessor.__init__().
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
                text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )

        headlines = document.meta["headlines"] if "headlines" in document.meta else []

        if clean_whitespace:
            text, headlines = self._clean_whitespace(text=text, headlines=headlines)

        if clean_empty_lines:
            text, headlines = self._clean_empty_lines(text=text, headlines=headlines)

        for substring in remove_substrings:
            text, _ = self._remove_substring(text=text, substring=substring, headlines=headlines)

        if text != document.content:
            document = deepcopy(document)
            document.content = text
        if headlines:
            document.meta["headlines"] = headlines

        return document

    def split(
        self,
        document: Union[dict, Document],
        split_by: Optional[Literal["word", "sentence", "passage"]],
        split_length: int,
        split_overlap: int,
        split_respect_sentence_boundary: bool,
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        """Perform document splitting on a single document. This method can split on different units, at different lengths,
        with different strides. It can also respect sentence boundaries. Its exact functionality is defined by
        the parameters passed into PreProcessor.__init__(). Takes a single document as input and returns a list of documents."""
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(document, dict):
            document = Document.from_dict(document, id_hash_keys=id_hash_keys)
        # Mainly needed for type checking
        if not isinstance(document, Document):
            raise HaystackError("Document must not be of type 'dict' but of type 'Document'.")

        if not split_by:
            return [document]

        if not split_length:
            raise Exception("split_length needs be set when using split_by.")

        if split_respect_sentence_boundary and split_by != "word":
            raise NotImplementedError("'split_respect_sentence_boundary=True' is only compatible with split_by='word'.")

        if type(document.content) is not str:
            logger.error("Document content is not of type str. Nothing to split.")
            return [document]

        text = document.content
        headlines = document.meta["headlines"] if "headlines" in document.meta else []

        if split_respect_sentence_boundary and split_by == "word":
            text_splits, splits_pages, splits_start_idxs = self._split_by_word_respecting_sent_boundary(
                text=text, split_length=split_length, split_overlap=split_overlap
            )
        else:
            # create individual "elements" of passage, sentence, or word
            elements, split_at = self._split_into_units(text=text, split_by=split_by)

            # concatenate individual elements based on split_length & split_stride
            text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
                elements=elements, split_length=split_length, split_overlap=split_overlap, split_at=split_at
            )

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
            for line in lines:
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
            if self.add_page_number:
                doc.meta["page"] = splits_pages[i]
            if headlines:
                split_start_idx = splits_start_idxs[i]
                relevant_headlines, earliest_rel_hl = self._extract_relevant_headlines_for_split(
                    headlines=headlines, split_txt=txt, split_start_idx=split_start_idx, earliest_rel_hl=earliest_rel_hl
                )
                doc.meta["headlines"] = relevant_headlines

            documents.append(doc)

        return documents

    @staticmethod
    def _extract_relevant_headlines_for_split(
        headlines: List[Dict], split_txt: str, split_start_idx: int, earliest_rel_hl: int
    ) -> Tuple[List[Dict], int]:
        """
        If you give it a list of headlines, a text split, and the start index of the split in the original text, this method
        extracts the headlines that are relevant for the split.
        """
        relevant_headlines = []

        for headline_idx in range(earliest_rel_hl, len(headlines)):
            # Headline is part of current split
            if split_start_idx <= headlines[headline_idx]["start_idx"] < split_start_idx + len(split_txt):
                headline_copy = deepcopy(headlines[headline_idx])
                headline_copy["start_idx"] = headlines[headline_idx]["start_idx"] - split_start_idx
                relevant_headlines.append(headline_copy)
            # Headline appears before current split, but might be relevant for current split
            elif headlines[headline_idx]["start_idx"] < split_start_idx:
                # Check if following headlines are on a higher level
                headline_to_check = headline_idx + 1
                headline_is_relevant = True
                while (
                    headline_to_check < len(headlines) and headlines[headline_to_check]["start_idx"] <= split_start_idx
                ):
                    if headlines[headline_to_check]["level"] <= headlines[headline_idx]["level"]:
                        headline_is_relevant = False
                        break
                    headline_to_check += 1
                if headline_is_relevant:
                    headline_copy = deepcopy(headlines[headline_idx])
                    headline_copy["start_idx"] = None
                    relevant_headlines.append(headline_copy)
                else:
                    earliest_rel_hl += 1
            # Headline (and all subsequent ones) only relevant for later splits
            elif headlines[headline_idx]["start_idx"] > split_start_idx + len(split_txt):
                break

        return relevant_headlines, earliest_rel_hl

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
