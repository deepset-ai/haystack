# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from haystack import logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install nltk>=3.9.1'") as nltk_imports:
    import nltk

logger = logging.getLogger(__name__)

Language = Literal[
    "ru", "sl", "es", "sv", "tr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "it", "no", "pl", "pt", "ml"
]

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

QUOTE_SPANS_RE = re.compile(r'"[^"]*"|\'[^\']*\'')

if nltk_imports.is_successful():

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
        nltk_imports.check()
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

        This method determines whether two adjacent sentence spans should be joined back together as a single sentence.
        It's used to prevent incorrect sentence splitting in specific cases like quotations, numbered lists,
        and parenthetical expressions.

        :param text: The text containing the spans.
        :param span: Tuple of (start, end) positions for the current sentence span.
        :param next_span: Tuple of (start, end) positions for the next sentence span.
        :param quote_spans: All quoted spans within text.
        :returns:
            True if the spans needs to be joined.
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
        if any(quote_start < end == quote_end and text[quote_end - 2] == "?" for quote_start, quote_end in quote_spans):
            # question is cited
            return True

        if re.search(r"(^|\n)\s*\d{1,2}\.$", text[start:end]) is not None:
            # sentence ends with a numeration
            return True

        # next sentence starts with a bracket or we return False
        return re.search(r"^\s*[\(\[]", text[next_start:next_end]) is not None

    @staticmethod
    def _read_abbreviations(lang: Language) -> List[str]:
        """
        Reads the abbreviations for a given language from the abbreviations file.

        :param lang: The language to read the abbreviations for.
        :returns: List of abbreviations.
        """
        abbreviations_file = Path(__file__).parent.parent.parent / f"data/abbreviations/{lang}.txt"
        if not abbreviations_file.exists():
            logger.warning("No abbreviations file found for {language}. Using default abbreviations.", language=lang)
            return []

        abbreviations = abbreviations_file.read_text().split("\n")
        return abbreviations
