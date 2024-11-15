# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from haystack import Document, component, logging
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.core.serialization import default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import serialize_callable

with LazyImport("Run 'pip install nltk'") as nltk_imports:
    import nltk

logger = logging.getLogger(__name__)

Language = Literal[
    "ru", "sl", "es", "sv", "tr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "it", "no", "pl", "pt", "ml"
]


@component
class NLTKDocumentSplitter(DocumentSplitter):
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        split_by: Literal["word", "sentence", "page", "passage", "function"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
        respect_sentence_boundary: bool = False,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Splits your documents using NLTK to respect sentence boundaries.

        Initialize the NLTKDocumentSplitter.

        :param split_by: Select the unit for splitting your documents. Choose from `word` for splitting by spaces (" "),
            `sentence` for splitting by NLTK sentence tokenizer, `page` for splitting by the form feed ("\\f") or
            `passage` for splitting by double line breaks ("\\n\\n").
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of overlapping units for each split.
        :param split_threshold: The minimum number of units per split. If a split has fewer units
            than the threshold, it's attached to the previous split.
        :param respect_sentence_boundary: Choose whether to respect sentence boundaries when splitting by "word".
            If True, uses NLTK to detect sentence boundaries, ensuring splits occur only between sentences.
        :param language: Choose the language for the NLTK tokenizer. The default is English ("en").
        :param use_split_rules: Choose whether to use additional split rules when splitting by `sentence`.
        :param extend_abbreviations: Choose whether to extend NLTK's PunktTokenizer abbreviations with a list
            of curated abbreviations, if available.
            This is currently supported for English ("en") and German ("de").
        :param splitting_function: Necessary when `split_by` is set to "function".
            This is a function which must accept a single `str` as input and return a `list` of `str` as output,
            representing the chunks after splitting.
        """

        super(NLTKDocumentSplitter, self).__init__(
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            split_threshold=split_threshold,
            splitting_function=splitting_function,
        )
        nltk_imports.check()
        if respect_sentence_boundary and split_by != "word":
            logger.warning(
                "The 'respect_sentence_boundary' option is only supported for `split_by='word'`. "
                "The option `respect_sentence_boundary` will be set to `False`."
            )
            respect_sentence_boundary = False
        self.respect_sentence_boundary = respect_sentence_boundary
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations
        self.sentence_splitter = SentenceSplitter(
            language=language,
            use_split_rules=use_split_rules,
            extend_abbreviations=extend_abbreviations,
            keep_white_spaces=True,
        )
        self.language = language

    def _split_into_units(
        self, text: str, split_by: Literal["function", "page", "passage", "sentence", "word", "line"]
    ) -> List[str]:
        """
        Splits the text into units based on the specified split_by parameter.

        :param text: The text to split.
        :param split_by: The unit to split the text by. Choose from "word", "sentence", "passage", or "page".
        :returns: A list of units.
        """

        if split_by == "page":
            self.split_at = "\f"
            units = text.split(self.split_at)
        elif split_by == "passage":
            self.split_at = "\n\n"
            units = text.split(self.split_at)
        elif split_by == "sentence":
            # whitespace is preserved while splitting text into sentences when using keep_white_spaces=True
            # so split_at is set to an empty string
            self.split_at = ""
            result = self.sentence_splitter.split_sentences(text)
            units = [sentence["sentence"] for sentence in result]
        elif split_by == "word":
            self.split_at = " "
            units = text.split(self.split_at)
        elif split_by == "function" and self.splitting_function is not None:
            return self.splitting_function(text)
        else:
            raise NotImplementedError(
                "DocumentSplitter only supports 'function', 'page', 'passage', 'sentence' or 'word' split_by options."
            )

        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += self.split_at
        return units

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split documents into smaller parts.

        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field source_id to track the original document.
                - A metadata field page_number to track the original page number.
                - All other metadata copied from the original document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            if self.respect_sentence_boundary:
                units = self._split_into_units(doc.content, "sentence")
                text_splits, splits_pages, splits_start_idxs = self._concatenate_sentences_based_on_word_amount(
                    sentences=units, split_length=self.split_length, split_overlap=self.split_overlap
                )
            else:
                units = self._split_into_units(doc.content, self.split_by)
                text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
                    elements=units,
                    split_length=self.split_length,
                    split_overlap=self.split_overlap,
                    split_threshold=self.split_threshold,
                )
            metadata = deepcopy(doc.meta)
            metadata["source_id"] = doc.id
            split_docs += self._create_docs_from_splits(
                text_splits=text_splits, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
            )
        return {"documents": split_docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        serialized = default_to_dict(
            self,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )
        if self.splitting_function:
            serialized["init_parameters"]["splitting_function"] = serialize_callable(self.splitting_function)
        return serialized

    @staticmethod
    def _number_of_sentences_to_keep(sentences: List[str], split_length: int, split_overlap: int) -> int:
        """
        Returns the number of sentences to keep in the next chunk based on the `split_overlap` and `split_length`.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: The number of sentences to keep in the next chunk.
        """
        # If the split_overlap is 0, we don't need to keep any sentences
        if split_overlap == 0:
            return 0

        num_sentences_to_keep = 0
        num_words = 0
        # Next overlapping Document should not start exactly the same as the previous one, so we skip the first sentence
        for sent in reversed(sentences[1:]):
            num_words += len(sent.split())
            # If the number of words is larger than the split_length then don't add any more sentences
            if num_words > split_length:
                break
            num_sentences_to_keep += 1
            if num_words > split_overlap:
                break
        return num_sentences_to_keep

    def _concatenate_sentences_based_on_word_amount(
        self, sentences: List[str], split_length: int, split_overlap: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Groups the sentences into chunks of `split_length` words while respecting sentence boundaries.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: A tuple containing the concatenated sentences, the start page numbers, and the start indices.
        """
        # Chunk information
        chunk_word_count = 0
        chunk_starting_page_number = 1
        chunk_start_idx = 0
        current_chunk: List[str] = []
        # Output lists
        split_start_page_numbers = []
        list_of_splits: List[List[str]] = []
        split_start_indices = []

        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            chunk_word_count += len(sentence.split())
            next_sentence_word_count = (
                len(sentences[sentence_idx + 1].split()) if sentence_idx < len(sentences) - 1 else 0
            )

            # Number of words in the current chunk plus the next sentence is larger than the split_length
            # or we reached the last sentence
            if (chunk_word_count + next_sentence_word_count) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = self._number_of_sentences_to_keep(
                    sentences=current_chunk, split_length=split_length, split_overlap=split_overlap
                )
                # Set up information for the new chunk
                if num_sentences_to_keep > 0:
                    # Processed sentences are the ones that are not overlapping with the next chunk
                    processed_sentences = current_chunk[:-num_sentences_to_keep]
                    chunk_starting_page_number += sum(sent.count("\f") for sent in processed_sentences)
                    chunk_start_idx += len("".join(processed_sentences))
                    # Next chunk starts with the sentences that were overlapping with the previous chunk
                    current_chunk = current_chunk[-num_sentences_to_keep:]
                    chunk_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    # Here processed_sentences is the same as current_chunk since there is no overlap
                    chunk_starting_page_number += sum(sent.count("\f") for sent in current_chunk)
                    chunk_start_idx += len("".join(current_chunk))
                    current_chunk = []
                    chunk_word_count = 0

        # Concatenate the sentences together within each split
        text_splits = []
        for split in list_of_splits:
            text = "".join(split)
            if len(text) > 0:
                text_splits.append(text)

        return text_splits, split_start_page_numbers, split_start_indices


if nltk_imports.is_successful():
    ISO639_TO_NLTK = {
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

    QUOTE_SPANS_RE = re.compile(r"\W(\"+|\'+).*?\1")

    class CustomPunktLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
        # The following adjustment of PunktSentenceTokenizer is inspired by:
        # https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
        # It is needed for preserving whitespace while splitting text into sentences.
        _period_context_fmt = r"""
            %(SentEndChars)s             # a potential sentence ending
            \s*                          # match potential whitespace [ \t\n\x0B\f\r]
            (?=(?P<after_tok>
                %(NonWord)s              # either other punctuation
                |
                (?P<next_tok>\S+)        # or some other token - original version: \s+(?P<next_tok>\S+)
            ))"""

        def period_context_re(self) -> re.Pattern:
            """
            Compiles and returns a regular expression to find contexts including possible sentence boundaries.

            :returns: A compiled regular expression pattern.
            """
            try:
                return self._re_period_context  # type: ignore
            except:  # noqa: E722
                self._re_period_context = re.compile(
                    self._period_context_fmt
                    % {
                        "NonWord": self._re_non_word_chars,
                        # SentEndChars might be followed by closing brackets, so we match them here.
                        "SentEndChars": self._re_sent_end_chars + r"[\)\]}]*",
                    },
                    re.UNICODE | re.VERBOSE,
                )
                return self._re_period_context

    def load_sentence_tokenizer(
        language: Language, keep_white_spaces: bool = False
    ) -> nltk.tokenize.punkt.PunktSentenceTokenizer:
        """
        Utility function to load the nltk sentence tokenizer.

        :param language: The language for the tokenizer.
        :param keep_white_spaces: If True, the tokenizer will keep white spaces between sentences.
        :returns: nltk sentence tokenizer.
        """
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except FileExistsError as error:
                logger.debug("NLTK punkt tokenizer seems to be already downloaded. Error message: {error}", error=error)

        language_name = ISO639_TO_NLTK.get(language)

        if language_name is not None:
            sentence_tokenizer = nltk.data.load(f"tokenizers/punkt_tab/{language_name}.pickle")
        else:
            logger.warning(
                "PreProcessor couldn't find the default sentence tokenizer model for {language}. "
                " Using English instead. You may train your own model and use the 'tokenizer_model_folder' parameter.",
                language=language,
            )
            sentence_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")

        if keep_white_spaces:
            sentence_tokenizer._lang_vars = CustomPunktLanguageVars()

        return sentence_tokenizer

    class SentenceSplitter:  # pylint: disable=too-few-public-methods
        """
        SentenceSplitter splits a text into sentences using the nltk sentence tokenizer
        """

        def __init__(
            self,
            language: Language = "en",
            use_split_rules: bool = True,
            extend_abbreviations: bool = True,
            keep_white_spaces: bool = False,
        ) -> None:
            """
            Initializes the SentenceSplitter with the specified language, split rules, and abbreviation handling.

            :param language: The language for the tokenizer. Default is "en".
            :param use_split_rules: If True, the additional split rules are used. If False, the rules are not used.
            :param extend_abbreviations: If True, the abbreviations used by NLTK's PunktTokenizer are extended by a list
                of curated abbreviations if available. If False, the default abbreviations are used.
                Currently supported languages are: en, de.
            :param keep_white_spaces: If True, the tokenizer will keep white spaces between sentences.
            """
            self.language = language
            self.sentence_tokenizer = load_sentence_tokenizer(language, keep_white_spaces=keep_white_spaces)
            self.use_split_rules = use_split_rules
            if extend_abbreviations:
                abbreviations = SentenceSplitter._read_abbreviations(language)
                self.sentence_tokenizer._params.abbrev_types.update(abbreviations)
            self.keep_white_spaces = keep_white_spaces

        def split_sentences(self, text: str) -> List[Dict[str, Any]]:
            """
            Splits a text into sentences including references to original char positions for each split.

            :param text: The text to split.
            :returns: list of sentences with positions.
            """
            sentence_spans = list(self.sentence_tokenizer.span_tokenize(text))
            if self.use_split_rules:
                sentence_spans = SentenceSplitter._apply_split_rules(text, sentence_spans)

            sentences = [{"sentence": text[start:end], "start": start, "end": end} for start, end in sentence_spans]
            return sentences

        @staticmethod
        def _apply_split_rules(text: str, sentence_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """
            Applies additional split rules to the sentence spans.

            :param text: The text to split.
            :param sentence_spans: The list of sentence spans to split.
            :returns: The list of sentence spans after applying the split rules.
            """
            new_sentence_spans = []
            quote_spans = [match.span() for match in QUOTE_SPANS_RE.finditer(text)]
            while sentence_spans:
                span = sentence_spans.pop(0)
                next_span = sentence_spans[0] if len(sentence_spans) > 0 else None
                while next_span and SentenceSplitter._needs_join(text, span, next_span, quote_spans):
                    sentence_spans.pop(0)
                    span = (span[0], next_span[1])
                    next_span = sentence_spans[0] if len(sentence_spans) > 0 else None
                start, end = span
                new_sentence_spans.append((start, end))
            return new_sentence_spans

        @staticmethod
        def _needs_join(
            text: str, span: Tuple[int, int], next_span: Tuple[int, int], quote_spans: List[Tuple[int, int]]
        ) -> bool:
            """
            Checks if the spans need to be joined as parts of one sentence.

            :param text: The text containing the spans.
            :param span: The current sentence span within text.
            :param next_span: The next sentence span within text.
            :param quote_spans: All quoted spans within text.
            :returns: True if the spans needs to be joined.
            """
            start, end = span
            next_start, next_end = next_span

            # sentence. sentence"\nsentence -> no split (end << quote_end)
            # sentence.", sentence -> no split (end < quote_end)
            # sentence?", sentence -> no split (end < quote_end)
            if any(quote_start < end < quote_end for quote_start, quote_end in quote_spans):
                # sentence boundary is inside a quote
                return True

            # sentence." sentence -> split (end == quote_end)
            # sentence?" sentence -> no split (end == quote_end)
            if any(
                quote_start < end == quote_end and text[quote_end - 2] == "?" for quote_start, quote_end in quote_spans
            ):
                # question is cited
                return True

            if re.search(r"(^|\n)\s*\d{1,2}\.$", text[start:end]) is not None:
                # sentence ends with a numeration
                return True

            # next sentence starts with a bracket or we return False
            return re.search(r"^\s*[\(\[]", text[next_start:next_end]) is not None

        @staticmethod
        def _read_abbreviations(language: Language) -> List[str]:
            """
            Reads the abbreviations for a given language from the abbreviations file.

            :param language: The language to read the abbreviations for.
            :returns: List of abbreviations.
            """
            abbreviations_file = Path(__file__).parent.parent / f"data/abbreviations/{language}.txt"
            if not abbreviations_file.exists():
                logger.warning(
                    "No abbreviations file found for {language}.Using default abbreviations.", language=language
                )
                return []

            abbreviations = abbreviations_file.read_text().split("\n")
            return abbreviations
