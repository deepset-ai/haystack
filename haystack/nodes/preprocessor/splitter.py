from typing import List, Optional, Tuple, Union

try:
    from typing import Literal, get_args
except ImportError:
    from typing_extensions import Literal, get_args  # type: ignore

import logging
import re
import warnings
from pathlib import Path
from copy import deepcopy
from pickle import UnpicklingError

import nltk
from tqdm.auto import tqdm
from nltk.tokenize.api import TokenizerI
from transformers import PreTrainedTokenizer

from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes.preprocessor.merger import DocumentMerger
from haystack.modeling.model.feature_extraction import FeatureExtractor


logger = logging.getLogger(__name__)


REGEX_METACHARS = r".^$*+?{}[]\|()"


SplitBy = Literal["character", "token", "word", "sentence", "paragraph", "page", "regex"]


class DocumentSplitter(BaseComponent):

    outgoing_edges = 1

    def __init__(
        self,
        split_by: SplitBy,
        split_length: int,
        split_regex: Optional[str] = None,
        split_overlap: int = 0,
        max_chars: int = 2000,
        tokenizer_model: Optional[Union[str, Path, PreTrainedTokenizer]] = None,
        nltk_language: str = "english",
        nltk_folder: Optional[str] = None,
        progress_bar: bool = True,
        add_page_number: bool = True,
    ):
        """
        Perform document splitting into smaller, shorter documents. Cn split on different units (word, sentence, etc),
        at different lengths, and include some overlap across the splits.
        It can also properly assign page numbers and re-assign headlines found in the metadata to each split document.

        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.
        :param split_regex: if split_by="regex", provide here a regex matching the separator. For example if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: Max. number of the above split unit (e.g. words) that are allowed in one document.
                             For instance, if n -> 10 & split_by -> "sentence", then each output document will contain 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                              For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                              `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                              Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param split_max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                                will cut the document, even mid-word, and log a loud error.\n
                                It's recommended to set this value approximately double double the size expect your documents
                                to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                                length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                                This is a safety parameter to avoid extremely long documents to end up in the document store.
                                Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                                performance of Reader nodes and might slow down drastically the indexing speed.
        :param tokenizer_model: If `split_by="token"`, you should provide a tokenizer model to compute the tokens,
                                for example `deepset/roberta-base-squad2`. You can give its identifier on HuggingFace Hub,
                                a local path to load it from, or an instance of PreTrainedTokenizer.
        :param nltk_language: The language used by "nltk.tokenize.sent_tokenize", for example "english", or "french".
                              Mind that some languages have limited support by the tokenizer: for example it seems incapable to split Chinese text
                              by word, but it can correctly split it by sentence.
        :param nltk_folder: Path to the folder containing the NTLK PunktSentenceTokenizer models, if loading a model from a local path.
                            Leave empty otherwise.
        :param progress_bar: Whether to show a progress bar.
        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.
        """
        super().__init__()
        self._validate_split_params(
            split_by=split_by, split_regex=split_regex, split_length=split_length, split_overlap=split_overlap
        )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        self.split_by = split_by
        self.split_regex = split_regex
        self.split_max_chars = max_chars
        self.progress_bar = progress_bar
        self.add_page_number = add_page_number

        self.merger = DocumentMerger(
            separator="",
            window_size=split_length,
            window_overlap=split_overlap,
            retain_page_number=True,
            realign_headlines=True,
        )

        self._nltk_language = nltk_language
        self._nltk_folder = Path(nltk_folder) if nltk_folder else None
        self.sentence_tokenizer = load_sentence_tokenizer(
            language=self._nltk_language, tokenizer_model_folder=self._nltk_folder
        )

        self.tokenizer = None
        if isinstance(tokenizer_model, PreTrainedTokenizer):
            self.tokenizer = tokenizer_model
        elif tokenizer_model:
            self.tokenizer = FeatureExtractor(pretrained_model_name_or_path=tokenizer_model)

        if split_by == "token" and not self.tokenizer:
            raise ValueError(
                "If you set split_by='token', you must give a value to 'tokenizer_model'. "
                "Use the same model you're using for your Reader."
            )

    def _validate_split_params(
        self, split_by: SplitBy, split_regex: Optional[str], split_length: int, split_overlap: int
    ):
        """
        Performs some basic validation on the parameters of the splitter.
        """
        if split_by not in get_args(SplitBy):
            raise ValueError(f"split_by must be one of: {', '.join(get_args(SplitBy))}")

        if not isinstance(split_length, int) or split_length <= 0:
            raise ValueError("split_length must be an integer > 0")

        if split_length:
            if not isinstance(split_overlap, int) or split_overlap < 0:
                raise ValueError("split_overlap must be an integer >= 0")

            if split_overlap >= split_length:
                raise ValueError("split_length must be higher than split_overlap")

        if split_regex and not split_by == "regex":
            logger.warning(
                "You provided a value to 'split_regex', but 'split_by=\"%s\"'. "
                "The document will be split by %s and the regex pattern will be ignored.",
                split_by,
                split_by,
            )

    @property
    def nltk_language(self):
        return self._nltk_language

    @nltk_language.setter
    def nltk_language(self, nltk_language):
        self.sentence_tokenizer = load_sentence_tokenizer(
            language=nltk_language, tokenizer_model_folder=self.nltk_folder
        )
        self._nltk_language = nltk_language

    @property
    def nltk_folder(self):
        return self._nltk_folder

    @nltk_folder.setter
    def nltk_folder(self, nltk_folder):
        self.sentence_tokenizer = load_sentence_tokenizer(
            language=self.nltk_language, tokenizer_model_folder=nltk_folder
        )
        self._nltk_folder = nltk_folder

    def run(  # type: ignore
        self,
        documents: List[Document],
        split_by: Optional[SplitBy] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_max_chars: Optional[int] = None,
        add_page_number: Optional[bool] = None,
    ):
        """
        Perform document splitting on a document. This method can split on different units, at different lengths,
        and include some overlap across the splits. It can also properly assign page numbers and re-assign headlines
        found in the metadata to each split document.

        No char should be lost in splitting, not even whitespace, and all headlines should be preserved.
        However, parts of the text and some headlines will be duplicated if `split_overlap > 0`.

        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page' or 'regex'.
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
        """
        split_by = split_by if split_by is not None else self.split_by
        split_regex = split_regex if split_regex is not None else self.split_regex
        split_length = split_length if split_length is not None else self.merger.window_size
        split_overlap = split_overlap if split_overlap is not None else self.merger.window_overlap
        split_max_chars = split_max_chars if split_max_chars is not None else self.split_max_chars
        add_page_number = add_page_number if add_page_number is not None else self.add_page_number

        self._validate_split_params(
            split_by=split_by, split_regex=split_regex, split_length=split_length, split_overlap=split_overlap
        )

        if split_by == "token" and not self.tokenizer:
            raise ValueError(
                "If you set split_by='token', you must give a value to 'tokenizer_model'. "
                "Use the same model you're using for your Reader."
            )

        if split_by == "character":
            splitter_function = lambda text: (text, [0] * len(text)) if text != "" else ([""], [0])

        elif split_by == "regex":
            if not split_regex:
                raise ValueError("If 'split_by' is set to 'regex', you must give a value to 'split_regex'.")
            else:
                splitter_function = lambda text: self.split_by_regex(text=text, pattern=split_regex)

        elif split_by == "page":
            splitter_function = lambda text: self.split_by_regex(text=text, pattern="\f")

        elif split_by == "paragraph":
            splitter_function = lambda text: self.split_by_regex(text=text, pattern="\n\n")

        elif split_by == "sentence":
            splitter_function = lambda text: self.split_by_sentence_tokenizer(text=text)

        elif split_by == "word":
            splitter_function = lambda text: self.split_by_regex(text=text, pattern="\s+")

        elif split_by == "token":
            splitter_function = lambda text: self.split_by_dense_tokenizer(text=text)

        else:
            raise ValueError("split_by must be either 'character', 'word', 'sentence', 'paragraph', 'page' or 'regex'")

        final_documents = []
        for document in documents:

            # Split them into single unit documents
            split_documents = self.split_into_units(
                document=document, units=splitter_function(text=document.content), add_page_number=add_page_number
            )[0]

            # Merge them back according to the given split_length and split_overlap, if needed
            if (split_length is not None and split_length > 1) or self.merger.window_size > 1:
                split_documents = self.merger.run(
                    documents=split_documents, window_size=split_length, window_overlap=split_overlap
                )[0]["documents"]

            # If a document longer than max_chars is found, split it into max_length chunks and log loudly.
            sane_documents = []
            for document in split_documents:
                if len(document.content) <= split_max_chars:
                    sane_documents.append(document)
                else:
                    logger.error(
                        "Found document with a character count higher than the maximum allowed (%s > %s). "
                        "The document is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                        "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.",
                        len(document.content),
                        split_max_chars,
                        split_max_chars,
                        len(document.content) - split_max_chars,
                    )
                    hard_splits = [
                        document.content[pos : pos + split_max_chars]
                        for pos in range(0, len(document.content), split_max_chars)
                    ]
                    sub_units = hard_splits, [0] * len(hard_splits)
                    sane_documents += self.split_into_units(
                        document=document,
                        units=sub_units,
                        add_page_number=add_page_number,
                        _start_from_page=document.meta.get("page", 1),
                    )[0]

            final_documents += sane_documents
        return {"documents": final_documents}, "output_1"

    def run_batch(  # type: ignore
        self,
        documents: List[List[Document]],
        split_by: Optional[SplitBy] = None,
        split_regex: Optional[str] = None,
        split_length: Optional[int] = None,
        split_overlap: Optional[int] = None,
        split_max_chars: Optional[int] = None,
        add_page_number: Optional[bool] = None,
    ):
        documents = [
            self.run(
                documents=docs,
                split_by=split_by,
                split_regex=split_regex,
                split_length=split_length,
                split_overlap=split_overlap,
                split_max_chars=split_max_chars,
                add_page_number=add_page_number,
            )[0]["documents"]
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Splitting", unit="docs")
        ]
        return {"documents": documents}, "output_1"

    def split_into_units(
        self,
        document: Document,
        units: Tuple[List[str], List[int]],
        add_page_number: bool = True,
        _start_from_page: int = 1,
    ) -> Tuple[List[Document], List[int]]:
        """
        Splits the parent document into single units documents.
        This algorithm is heavily simplified by the lack of split_length and split_overlap.
        Those are later applied by the merger.
        """
        if isinstance(document, dict):
            warnings.warn(
                "Passing a dictionary to Preprocessor.split() is deprecated. Use Document objects.", DeprecationWarning
            )
            document = Document.from_dict(document)

        if document.content_type != "text":
            raise ValueError(
                f"Document content type is not 'text', but '{document.content_type}'. Preprocessor only handles text documents."
            )

        headlines_to_assign = deepcopy(document.meta.get("headlines", [])) or []
        unit_documents = []
        pages = _start_from_page
        position_in_document = 0
        for text in units[0]:

            # Find the relevant headlines for this unit
            unit_headlines = []
            other_headlines = []
            for headline in headlines_to_assign:
                if position_in_document <= headline["start_idx"] < position_in_document + len(text):
                    headline["start_idx"] -= position_in_document
                    unit_headlines.append(headline)
                else:
                    other_headlines.append(headline)

            position_in_document += len(text)
            headlines_to_assign = other_headlines

            # Clone the meta from the parent document
            unit_meta = deepcopy(document.meta)

            # If the parent had headlines, but this unit happens not to have them, we assign an empty list
            # If the parent never had a headlines field, we don't create it here.
            if "headlines" in unit_meta and unit_meta["headlines"]:
                unit_meta["headlines"] = unit_headlines

            # Assing page number if required
            if "page" in unit_meta:
                del unit_meta["page"]
            if add_page_number:
                unit_meta["page"] = pages
                pages += text.count("\f")

            # Create the document
            unit_document = Document(content=text, meta=unit_meta, id_hash_keys=document.id_hash_keys)
            unit_documents.append(unit_document)

        return unit_documents, units[1]

    @staticmethod
    def split_by_regex(pattern: str, text: str) -> Tuple[List[str], List[int]]:
        """
        Split a long text into chunks based on a regex match.

        :param splitter: the text, or regex, to split the text upon
        :param text: the text to split
        :return: the list of splits, along with the lenght of the separators matched.
        """
        matches = [(match.start(), match.end()) for match in re.compile(pattern).finditer(text)]
        if not matches:
            return [text], [0]

        if matches and not matches[-1][1] == len(text):
            matches.append((len(text), len(text)))

        units = []
        offsets = []
        for start_match, end_match in zip([(None, 0), *matches[:-1]], matches):
            units.append(text[start_match[1] : end_match[1]])
            offsets.append(end_match[1] - end_match[0])

        return units, offsets

    def split_by_sentence_tokenizer(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Splits a given text with an NLTK sentence tokenizer, preserving all whitespace.

        :param text: the text to tokenize
        :return the tokenized text as a list of strings
        """
        token_spans = self.sentence_tokenizer.span_tokenize(text)
        return split_on_spans(text=text, spans=token_spans)

    def split_by_dense_tokenizer(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Splits a given text with a tokenizer, preserving all whitespace.

        :param text: the text to tokenize
        :return the tokenized text as a list of strings
        """
        token_batch = self.tokenizer(text=text)
        token_positions = [pos for pos, i in enumerate(token_batch.sequence_ids()) if i == 0]
        token_chars = [token_batch.token_to_chars(i) for i in token_positions]
        token_spans = [(chars.start, chars.end) for chars in token_chars]
        return split_on_spans(text=text, spans=token_spans)


def split_on_spans(text: str, spans: List[Tuple[int, int]]) -> Tuple[List[str], List[int]]:
    """
    Splits a given text on the arbitrary spans given.

    :param text: the text to tokenize
    :param spans: the spans to split on.
    :return the tokenized text as a list of strings
    """
    units = []
    prev_token_start = 0
    for token_start, _ in spans:

        if prev_token_start != token_start:
            units.append(text[prev_token_start:token_start])
            prev_token_start = token_start

    if prev_token_start != len(text):
        units.append(text[prev_token_start:])

    if not units:
        return [text], [0]
    return units, [0] * len(units)


def load_sentence_tokenizer(
    language: Optional[str] = None, tokenizer_model_folder: Optional[Path] = None
) -> TokenizerI:
    """
    Attempt to load the sentence tokenizer with sensible fallbacks.

    Tried to load from self.tokenizer_model_folder first, then falls back to 'tokenizers/punkt' and eventually
    falls back to the default English tokenizer.
    """
    # Try loading from the specified path
    if tokenizer_model_folder and language:
        tokenizer_model_path = Path(tokenizer_model_folder) / f"{language}.pickle"
        try:
            return nltk.data.load(f"file:{str(tokenizer_model_path)}", format="pickle")
        except LookupError as e:
            logger.exception(f"PreProcessor couldn't load sentence tokenizer from {tokenizer_model_path}")
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                f"PreProcessor couldn't find custom sentence tokenizer model for {language} in {tokenizer_model_folder}. "
            )
        logger.warning("Trying to find a tokenizer for %s under the default path (tokenizers/punkt/).", language)

    # Try loading from the default path
    if language:
        try:
            return nltk.data.load(f"tokenizers/punkt/{language}.pickle")
        except LookupError as e:
            logger.exception(
                "PreProcessor couldn't load sentence tokenizer from the default tokenizer path (tokenizers/punkt/)"
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "PreProcessor couldn't find custom sentence tokenizer model for %s in the default tokenizer path (tokenizers/punkt/)",
                language,
            )
        logger.warning(
            "Using an English tokenizer as fallback. You may train your own model and use the 'tokenizer_model_folder' parameter."
        )

    # Fallback to English from the default path
    return nltk.data.load(f"tokenizers/punkt/english.pickle")
