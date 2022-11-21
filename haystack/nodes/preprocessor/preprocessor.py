from typing import List, Optional, Tuple, Iterable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import logging
import re
from copy import deepcopy
from itertools import accumulate
import warnings
from pathlib import Path
from pickle import UnpicklingError

import nltk
from tqdm.auto import tqdm
from nltk.tokenize import NLTKWordTokenizer
from nltk.tokenize.api import TokenizerI

from haystack.nodes import BaseComponent
from haystack.schema import Document


logger = logging.getLogger(__name__)


REGEX_METACHARS = r".^$*+?{}[]\|()"


class PreProcessor(BaseComponent):
    def __init__(
        self,
        split_by: Literal["word", "sentence", "paragraph", "page", "regex"],
        split_length: int,
        clean_whitespace: bool,
        clean_empty_lines: bool,
        clean_header_footer: bool,
        clean_regex: Optional[str] = None,
        header_footer_n_chars: int = 50,
        header_footer_pages_to_ignore: List[int] = [],
        split_regex: Optional[str] = None,
        split_overlap: int = 0,
        language: str = "english",
        tokenizer_model_folder: Optional[str] = None,
        progress_bar: bool = True,
        add_page_number: bool = False,
        clean_substrings: List[str] = [],
        split_respect_sentence_boundary: Optional[bool] = None,
    ):
        """
        Perform document cleaning and splitting.

        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param clean_regex: Remove the specified regex matches from the text. For example, `clean_regex='[0-9]'`
                            removes all digits from the document's content, and `clean_regex='(a string|another string)'`
                            will remove all occurrences of either string from the document content.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                    for the longest common string. This heuristic uses exact matches and therefore
                                    works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4"
                                    or similar. Use 'clean_regex' to detect such headers.
        :param header_footer_n_chars: how many chars to look for headers and footer in. Defautls to 50.
        :param header_footer_pages_to_ignore: which pages to ignore in the header-footer detection heuristic.
                                              Takes a list of pages numbers. It ignores no pages by default.

        :param split_by: Unit for splitting the document. Can be "word", "sentence", "paragraph", "page", "regex".
        :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document.
                             For instance, if n -> 10 & split_by -> "sentence", then each output document will contain 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                              For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                              `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.

        :param language: The language used by "nltk.tokenize.sent_tokenize", for example "english", or "french".
                         Mind that some languages have limited support by the tokenizer: for example it seems incapable to split Chinese text
                         by word, but it can correctly split it by sentence.
        :param tokenizer_model_folder: Path to the folder containing the NTLK PunktSentenceTokenizer models, if loading a model from a local path.
                                       Leave empty otherwise.
        :param progress_bar: Whether to show a progress bar.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.

        :param split_respect_sentence_boundary: deprecated, use 'split_by="sentence"' and adjust the value of 'split_length'.
        :param clean_substrings: Deprecated, use 'clean_regex' as 'clean_regex=r"(first substring|second substring|third substring)"'.
        """
        super().__init__()

        if not isinstance(header_footer_n_chars, int) or header_footer_n_chars < 0:
            raise ValueError("header_footer_n_chars must be an integer >= 0")

        if not isinstance(header_footer_pages_to_ignore, Iterable) or any(
            not isinstance(page, int) or page < 0 for page in header_footer_pages_to_ignore
        ):
            raise ValueError("header_footer_pages_to_ignore should be an iterable containing only integers >= 0")

        if not isinstance(split_length, int) or split_length < 1:
            raise ValueError("split_length must be an integer >= 1")

        if not isinstance(split_overlap, int) or split_overlap < 0:
            raise ValueError("split_overlap must be an integer >= 0")

        if split_overlap >= split_length:
            raise ValueError("split_length must be higher than split_overlap")

        if not tokenizer_model_folder:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

        self.clean_whitespace = clean_whitespace
        self.clean_empty_lines = clean_empty_lines
        self.clean_regex = clean_regex
        self.clean_header_footer = clean_header_footer
        self.header_footer_n_chars = header_footer_n_chars
        self.header_footer_pages_to_ignore = header_footer_pages_to_ignore or []

        self.split_by = split_by
        self.split_regex = split_regex
        self.split_length = split_length
        self.split_overlap = split_overlap

        self._language = language
        self._tokenizer_model_folder = tokenizer_model_folder
        self.tokenizer = load_tokenizer(language_name=language, tokenizer_model_folder=tokenizer_model_folder)

        self.progress_bar = progress_bar
        self.add_page_number = add_page_number

        if split_respect_sentence_boundary is not None:
            warnings.warn(
                "'split_respect_sentence_boundary' is deprecated. "
                "Setting 'split_by=\"word\"', sentence boundaries are never respected. "
                "Use 'split_by=\"sentence\"' to have the sentence boundaries respected. "
                "However, keep in mind that the 'split_length' will need to be adjusted, "
                "as it now refers to the number of chars.",
                DeprecationWarning,
            )
            self.split_respect_sentence_boundary = split_respect_sentence_boundary

        if clean_substrings:
            warnings.warn("clean_substrings is deprecated, use 'clean_regex'.", DeprecationWarning)
            if self.clean_regex:
                logger.warning("'clean_substrings' will be overwritten by the value of 'clean_regex'.")
            else:
                clean_substrings = [
                    substring.replace(char, "\\" + char) for substring in clean_substrings for char in REGEX_METACHARS
                ]
                self.clean_regex = f"({'|'.join(clean_substrings)})"

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, language):
        self.tokenizer = load_tokenizer(language_name=language, tokenizer_model_folder=self.tokenizer_model_folder)
        self._language = language

    @property
    def tokenizer_model_folder(self):
        return self._tokenizer_model_folder

    @tokenizer_model_folder.setter
    def tokenizer_model_folder(self, tokenizer_model_folder):
        self.tokenizer = load_tokenizer(language_name=self.language, tokenizer_model_folder=tokenizer_model_folder)
        self._tokenizer_model_folder = tokenizer_model_folder

    def run(  # type: ignore
        self,
        documents: List[Document],
        clean_whitespace: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        clean_header_footer: Optional[bool] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
        split_by: Literal["word", "sentence", "paragraph", "page", "regex"] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        add_page_number: Optional[bool] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        clean_substrings: Optional[List[str]] = None,
    ):
        """
        :param clean_whitespace: Strip whitespaces before or after each line in the text.
        :param clean_empty_lines: Remove more than two empty lines in the text.
        :param clean_regex: Remove the specified regex matches from the text. For example, `clean_regex='[0-9]'`
                            removes all digits from the document's content, and `clean_regex='(a string|another string)'`
                            will remove all occurrences of either string from the document content.
        :param clean_header_footer: Use heuristic to remove footers and headers across different pages by searching
                                     for the longest common string. This heuristic uses exact matches and therefore
                                     works well for footers like "Copyright 2019 by The Author", but won't detect "Page 3 of 4"
                                     or similar.
        :param header_footer_n_chars: how many chars to look for headers and footer in. Defautls to 50.
        :param header_footer_pages_to_ignore: which pages to ignore in the header-footer detection heuristic.
                                              Takes a list of pages numbers. It ignores no pages by default.

        :param split_by: Unit for splitting the document. Can be "word", "sentence", "paragraph", "page", "regex".
        :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document. For instance,
                             if `split_length=10 and split_by="sentence"`, then each output document will have 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                              For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                              `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.

        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.

        :param split_respect_sentence_boundary: deprecated, use 'split_by="sentence"' and adjust the value of 'split_length'.
        :param clean_substrings: Deprecated, use 'clean_regex' as 'clean_regex=r"(first substring|second substring|third substring)"'.
        """
        if isinstance(documents, Document):
            warnings.warn(
                "Passing single documents to Preprocessor.process() is deprecated. Pass a list of Document objects",
                DeprecationWarning,
            )
            documents = [documents]

        elif isinstance(documents, dict):
            warnings.warn(
                "Passing dictionaries to Preprocessor.process() is deprecated. Pass a list of Document objects.",
                DeprecationWarning,
            )
            documents = [Document.from_dict(documents)]

        elif isinstance(documents, list) and isinstance(documents[0], dict):
            warnings.warn(
                "Passing dictionaries to Preprocessor.process() is deprecated. Pass a list of Document objects.",
                DeprecationWarning,
            )
            documents = [Document.from_dict(doc) for doc in documents]

        elif not isinstance(documents, list) or not isinstance(documents[0], Document):
            raise ValueError("'documents' must be a list of Document objects.")

        elif any(document.content_type != "text" for document in documents):
            ids = [doc.id for doc in documents if doc.content_type != "text"]
            raise ValueError(
                "Documents list contains one or more documents that are not of type 'text' "
                f"(doc ids: '{', '.join(ids)}'). Preprocessor only handles text documents."
            )

        if clean_whitespace is None:
            clean_whitespace = self.clean_whitespace
        if clean_header_footer is None:
            clean_header_footer = self.clean_header_footer
        if clean_empty_lines is None:
            clean_empty_lines = self.clean_empty_lines
        if header_footer_n_chars is None:
            header_footer_n_chars = self.header_footer_n_chars
        if header_footer_pages_to_ignore is None:
            header_footer_pages_to_ignore = self.header_footer_pages_to_ignore

        if split_by is None:
            split_by = self.split_by
        if split_regex is None:
            split_regex = self.split_regex
        if split_length is None:
            split_length = self.split_length
        if split_overlap is None:
            split_overlap = self.split_overlap
        if split_respect_sentence_boundary is None:
            split_respect_sentence_boundary = self.split_respect_sentence_boundary

        # Small optimization
        if (
            split_by == "regex"
            and clean_regex == split_regex
            and not clean_whitespace
            and not clean_empty_lines
            and not clean_header_footer
        ):
            split_docs = [
                split(
                    document=doc,
                    split_by=split_by,
                    split_regex=split_regex,
                    split_length=split_length,
                    split_overlap=split_overlap,
                    add_page_number=add_page_number,
                    tokenizer=self.tokenizer,
                    _clean_separator=True,
                )
                for doc in documents
            ]
            documents = [doc for docs in split_docs for doc in docs]
            return {"documents": documents}, "output_1"

        documents = [
            clean(
                document=doc,
                clean_whitespace=clean_whitespace,
                clean_empty_lines=clean_empty_lines,
                clean_regex=clean_regex,
                clean_header_footer=clean_header_footer,
                header_footer_n_chars=header_footer_n_chars,
                header_footer_pages_to_ignore=header_footer_pages_to_ignore,
                clean_substrings=clean_substrings,
            )
            for doc in documents
        ]
        split_docs = [
            split(
                document=doc,
                split_by=split_by,
                split_regex=split_regex,
                split_length=split_length,
                split_overlap=split_overlap,
                add_page_number=add_page_number,
                tokenizer=self.tokenizer,
            )
            for doc in documents
        ]
        documents = [doc for docs in split_docs for doc in docs]
        return {"documents": documents}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        clean_whitespace: Optional[bool] = None,
        clean_empty_lines: Optional[bool] = None,
        clean_regex: Optional[str] = None,
        clean_header_footer: Optional[bool] = None,
        header_footer_n_chars: Optional[int] = None,
        header_footer_pages_to_ignore: Optional[List[int]] = None,
        split_by: Literal["word", "sentence", "paragraph", "page", "regex"] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        add_page_number: Optional[bool] = None,
        split_respect_sentence_boundary: Optional[bool] = None,
        clean_substrings: Optional[List[str]] = None,
    ):
        nested_docs = [
            self.run(
                documents=docs,
                clean_whitespace=clean_whitespace,
                clean_empty_lines=clean_empty_lines,
                clean_regex=clean_regex,
                clean_header_footer=clean_header_footer,
                header_footer_n_chars=header_footer_n_chars,
                header_footer_pages_to_ignore=header_footer_pages_to_ignore,
                split_by=split_by,
                split_regex=split_regex,
                split_length=split_length,
                split_overlap=split_overlap,
                add_page_number=add_page_number,
                clean_substrings=clean_substrings,
                split_respect_sentence_boundary=split_respect_sentence_boundary,
            )
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Preprocessing", unit="docs")
        ]
        return {"documents": nested_docs}, "output_1"


def split(
    document: Document,
    split_by: Literal["word", "sentence", "paragraph", "page", "regex"],
    split_regex: Optional[str] = None,
    split_length: int = 1,
    split_overlap: int = 0,
    split_max_chars: int = 2000,
    add_page_number: bool = None,
    tokenizer: TokenizerI = None,
    _clean_separator: bool = False,
) -> List[Document]:
    """
    Perform document splitting on a document. This method can split on different units, at different lengths,
    and include some overlap across the splits. It can also properly assign page numbers and re-assign headlines
    found in the metadata to each split document.

    No char should be lost in splitting, not even whitespace, and all headlines should be preserved.
    However, parts of the text and some headlines will be duplicated if `split_overlap > 0`.

    :param split_by: Unit for splitting the document. Can be "word", "sentence", "paragraph", "page" or "regex".
    :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                        should be split on "--my separator--", this field should be `splitter="--my separator--"`
    :param split_length: Max. number of units (words, sentences, paragraph or pages, according to split_by)
                            that are allowed in one document.
    :param split_overlap: Unit overlap between two adjacent documents after a split.
                            Setting this to a positive number essentially enables the sliding window approach.
                            Set the value to 0 to ensure there is no overlap among the documents after splitting.
    :param split_max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            will cut the document, even mid-word, and log a loud error.\n
                            It's recommended to set this value approximately double double the size expect your documents
                            to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                            length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                            This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and might slow down drastically the indexing speed.
    :param add_page_number: Saves in the metadata ('page' key) the page number where the document content comes from.
    :param tokenizer: tokenizers can be language specific. If splitting by word or sentence, pass a specific tokenizer
                      loaded by `load_tokenizer` to ensure a proper splitting depending on the language on your text.
                      This is especially important for languages with non-Latin punctuation and word boundaries,
                      like Chinese or Arabic.
    :param _clean_separator: Whether to keep the separator in case of split_by regex, page or paragraph. Internal, don't use.
    """

    if split_overlap >= split_length:
        raise ValueError(f"split_length ({split_length}) must be greater than split_overlap ({split_overlap})")

    if document.content_type != "text":
        raise ValueError(
            f"Document content type is not 'text', but '{document.content_type}'. Preprocessor only handles text documents."
        )

    if split_regex and not split_by == "regex":
        logger.warning(
            "You provided a value to 'split_regex', but 'split_by=\"%s\"'. "
            "The document will be split by %s and the regex pattern will be ignored.",
            split_by,
            split_by,
        )

    if split_by == "regex":
        if not split_regex:
            raise ValueError("If 'split_by' is set to 'regex', you must give a value to 'split_regex'.")
        units, offsets = split_by_regex(text=document.content, pattern=split_regex, _clean_separator=_clean_separator)

    elif split_by == "page":
        units, offsets = split_by_regex(text=document.content, pattern="\f")

    elif split_by == "paragraph":
        units, offsets = split_by_regex(text=document.content, pattern="\n\n")

    elif split_by == "sentence":
        tokenizer = tokenizer or load_tokenizer()
        units = split_by_tokenizer(text=document.content, tokenizer=tokenizer)

    elif split_by == "word":
        units = split_by_tokenizer(text=document.content, tokenizer=NLTKWordTokenizer())

    else:
        raise ValueError("split_by must be either 'word', 'sentence', 'paragraph', 'page' or 'regex'")

    if not _clean_separator:
        offsets = [0] * len(units)

    # Create the groups according to split_lenght and split_overlap
    positions = [0] + list(accumulate([len(unit) + offset for unit, offset in zip(units, offsets)]))
    splits = [
        (
            "".join(units[pos : pos + split_length]),  # The split's text
            positions[pos],  # The split's starting character position in the source document
        )
        for pos in range(0, max(1, len(units) - split_overlap), split_length - split_overlap)
    ]

    # Headlines MUST be sorted by start_idx
    if document.meta.get("headlines"):
        document.meta["headlines"] = sorted(document.meta["headlines"], key=lambda h: h["start_idx"])

    split_documents = _split_document(
        document=document, splits=splits, max_chars=split_max_chars, add_page_number=add_page_number
    )
    return split_documents


def _split_document(
    document: Document, splits: List[Tuple[str, int]], max_chars: int, add_page_number: bool = True
) -> List[Document]:
    """
    Accessory function that splits a large document into the given chunks.
    Deals with too long chunks by splitting them over max_chars.
    Takes care of calculating page numbers and including only the relevant headlines into each smaller document.

    :param document: the original document the chunks come from. Needed to clone things such as id_hash_keys
    :param splits: a list of tuple containing the chunks of text this document will be split into and their
                    original position in the original document's content.
    :param max_chars: the maximum lenght in chars that the new small documents will have. If surpassed, the
                        affected document will be hard-split on the last char and the function will log loudly.
    :param add_page_number: whether to compute the page number for the new documents.
    :raturn: a list of small documents, each never longer than max_chars.
    """
    headlines = deepcopy(document.meta.get("headlines", []))
    split_docs = []
    page_index = document.meta.get("page", 1) - 1  # 0

    # Page number must be tracked separately due to the split_overlap
    if add_page_number:
        page_start_positions = ([-1] * page_index) + [pos for pos, char in enumerate(document.content) if char == "\f"]

    for split, position_in_document in splits:
        split_doc = Document(content=split, meta=deepcopy(document.meta), id_hash_keys=document.id_hash_keys)

        # See how many pages we crossed in this chunk
        if add_page_number:
            while page_index < len(page_start_positions) and page_start_positions[page_index] < position_in_document:
                page_index += 1
            split_doc.meta["page"] = page_index + 1

        # Find all the headlines starting in this chunk (we don't account for headlines that are split across documents)
        split_doc_headlines = []
        while headlines and headlines[0]["start_idx"] < position_in_document:
            headlines = headlines[1:]
        if headlines:
            for headline in headlines:
                if headline["start_idx"] < position_in_document + len(split):
                    split_doc_headlines.append({**headline, "start_idx": headline["start_idx"] - position_in_document})
        split_doc.meta["headlines"] = split_doc_headlines

        # If a document longer than max_chars is found, split it into max_length chunks and log loudly.
        if len(split) <= max_chars:
            split_docs.append(split_doc)
        else:
            logger.error(
                "Found document with a character count higher than the maximum allowed (%s > %s). "
                "The document is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                "Keep in mind that very long Documents can severely impact the performance of Readers.",
                len(split),
                max_chars,
                max_chars,
                len(split) - max_chars,
            )
            hard_splits = [(split[pos : pos + max_chars], pos) for pos in range(0, len(split), max_chars)]
            split_docs += _split_document(
                document=split_doc, splits=hard_splits, max_chars=max_chars, add_page_number=add_page_number
            )

    return split_docs


def split_by_regex(pattern: str, text: str, _clean_separator: bool = False) -> Tuple[List[str], List[int]]:
    """
    Split a long text into chunks based on a regex match.

    :param splitter: the text, or regex, to split the text upon
    :param text: the text to split
    :param _clean_separator: internal, don't use
    :return: the list of splits, along with the lenght of the removed separators.
    """
    matches = [(match.start(), match.end()) for match in re.compile(pattern).finditer(text)]
    if not matches:
        return [text], []

    if matches and not matches[-1][1] == len(text):
        matches.append((len(text), len(text)))

    units = []
    if _clean_separator:
        separators = [match[1] - match[0] for match in matches]
        for start_match, end_match in zip([(None, 0), *matches[:-1]], matches):
            units.append(text[start_match[1] : end_match[0]])

    else:
        separators = [0] * len(units)
        for start_match, end_match in zip([(None, 0), *matches[:-1]], matches):
            units.append(text[start_match[1] : end_match[1]])

    return units, separators


def split_by_tokenizer(text: str, tokenizer: TokenizerI) -> List[str]:
    """
    Splits a given text into tokens, preserving all whitespace.

    :param text: the text to tokenize
    :param tokenizer: the tokenizer to use (might be a sentence or word tokenizer)
    :return the tokenized text as a list of strings
    """
    token_spans = tokenizer.span_tokenize(text)

    units = []
    prev_token_start = 0
    for token_start, _ in token_spans:

        if prev_token_start != token_start:
            units.append(text[prev_token_start:token_start])
            prev_token_start = token_start

    if prev_token_start != len(text):
        units.append(text[prev_token_start:])

    return units


def load_tokenizer(
    language_name: Optional[str], tokenizer_model_folder: Optional[Path] = None
) -> nltk.tokenize.punkt.PunktSentenceTokenizer:
    """
    Attempt to load the sentence tokenizer with sensible fallbacks.

    Tried to load from self.tokenizer_model_folder first, then falls back to 'tokenizers/punkt' and eventually
    falls back to the default English tokenizer.
    """
    # Try loading from the specified path
    if tokenizer_model_folder and language_name:
        tokenizer_model_path = Path(tokenizer_model_folder) / f"{language_name}.pickle"
        try:
            return nltk.data.load(f"file:{str(tokenizer_model_path)}", format="pickle")
        except LookupError as e:
            logger.exception(f"PreProcessor couldn't load sentence tokenizer from {tokenizer_model_path}")
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                f"PreProcessor couldn't find custom sentence tokenizer model for {language_name} in {tokenizer_model_folder}. "
            )
        logger.warning("Trying to find a tokenizer for %s under the default path (tokenizers/punkt/).", language_name)

    # Try loading from the default path
    if language_name:
        try:
            return nltk.data.load(f"tokenizers/punkt/{language_name}.pickle")
        except LookupError as e:
            logger.exception(
                "PreProcessor couldn't load sentence tokenizer from the default tokenizer path (tokenizers/punkt/)"
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "PreProcessor couldn't find custom sentence tokenizer model for %s in the default tokenizer path (tokenizers/punkt/)",
                language_name,
            )
        logger.warning(
            "Using an English tokenizer as fallback. You may train your own model and use the 'tokenizer_model_folder' parameter."
        )

    # Fallback to English from the default path
    return nltk.data.load(f"tokenizers/punkt/english.pickle")


def clean(
    document: Document,
    clean_whitespace: bool,
    clean_header_footer: bool,
    clean_empty_lines: bool,
    clean_substrings: Optional[List[str]] = None,
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
    :param clean_substrings: Deprecated, use `clean_regex`
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

    if clean_substrings:
        warnings.warn("clean_substrings is deprecated, use clean_regex", DeprecationWarning)
        clean_regex = f"({'|'.join(clean_substrings)})"

    if not isinstance(header_footer_n_chars, int) or header_footer_n_chars < 0:
        raise ValueError("header_footer_n_chars must be an integer >= 0")

    if any(not isinstance(page, int) or page < 0 for page in header_footer_pages_to_ignore):
        raise ValueError("header_footer_pages_to_ignore must contain only integers >= 0")

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
        clean_document = replace_regex_matches(document=document, pattern=r"[ \t\r\v]*\f[ \t\r\v]*", string="\f")
        # Whitespace around newlines
        clean_document = replace_regex_matches(document=document, pattern=r"[ \t\r\v]*\n[ \t\r\v]*", string="\n")
        # Leading and trailing spaces
        clean_document = replace_regex_matches(document=document, pattern=r"^[ \t\r\v]*", string="")
        clean_document = replace_regex_matches(document=document, pattern=r"[ \t\r\v]*$", string="")

    if clean_empty_lines:
        clean_document = replace_regex_matches(document=document, pattern=r"[\n]{2,}", string="\n")

    if clean_regex:
        clean_document = replace_regex_matches(document=document, pattern=clean_regex, string="")

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

    split_docs = split(split_by="regex", split_regex=pattern, document=document, _clean_separator=True)
    new_content = string.join([doc.content for doc in split_docs])

    # check for a trailing match that might have been left out in the above cleanup
    trailing_match = re.compile(rf"{pattern}$").search(document.content)
    if trailing_match:
        new_content += string

    # Re-align headlines
    pos_in_new_doc = 0
    new_headlines = []
    compiled_pattern = re.compile(pattern)
    for doc in split_docs:
        for headline in doc.meta.get("headlines", []):
            # If the headline contains the pattern to remove, take it out
            headline["content"] = compiled_pattern.sub(string, headline["content"])
            if headline["content"]:  # Some headlines might get fully erased
                headline["start_idx"] += pos_in_new_doc
                new_headlines.append(headline)
        pos_in_new_doc += len(doc.content) + len(string)

    document.content = new_content
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
