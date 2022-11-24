from typing import List, Optional, Tuple, Union, Literal

try:
    from typing import Literal, get_args
except ImportError:
    from typing_extensions import Literal, get_args  # type: ignore

import math
import logging
import re
import warnings
from enum import Enum
from math import inf
from pathlib import Path
from copy import deepcopy
from pickle import UnpicklingError

import nltk
from tqdm.auto import tqdm
from nltk import NLTKWordTokenizer
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
    """
    Splits documents into smaller, shorter documents.

    Can split on different units ('character', 'token', 'word', 'sentence', 'paragraph', 'page', or 'regex'),
    at different lengths, and include some overlap across the splits.

    It can also properly assign page numbers and re-assign headlines found in the metadata of the parent document
    to each split document.

    No char is lost in splitting, not even whitespace, and all headlines are preserved, However, text and headlines
    duplication may occur if `split_overlap>0`.
    """

    outgoing_edges = 1

    def __init__(
        self,
        split_by: SplitBy,
        split_length: int,
        split_regex: Optional[str] = None,
        split_overlap: int = 0,
        max_chars: int = 2000,
        max_tokens: int = 0,
        tokenizer_model: Optional[Union[str, Path, PreTrainedTokenizer]] = None,
        nltk_language: str = "english",
        nltk_folder: Optional[str] = None,
        progress_bar: bool = True,
        add_page_number: bool = True,
    ):
        """
        Splits documents into smaller, shorter documents.

        Can split on different units ('character', 'token', 'word', 'sentence', 'paragraph', 'page', or 'regex'),
        at different lengths, and include some overlap across the splits.

        It can also properly assign page numbers and re-assign headlines found in the metadata of the parent document
        to each split document.

        No char is lost in splitting, not even whitespace, and all headlines are preserved, However, text and headlines
        duplication may occur if `split_overlap>0`.

        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.

        :param split_regex: If `split_by="regex"`, provide here a regex matching the separator. For example, if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.

        :param split_length: The maximum number of the above split unit (like word, sentence, page and so on) that are allowed in one document.
                                For instance, if `split_lenght=10` and `split_by="sentence"`, then each output document will contain 10 sentences.\n
                                Note that split_length can be set to 0 to mean "infinite". This can be useful with `max_tokens`.

        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                                For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                                `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                                Set the value to 0 to ensure there is no overlap among the documents after splitting.

        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                            length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                            This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.

        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it requires `tokenizer_model` to be set to the
                            model of your Reader and will verify that, whatever your `split_length` value is set
                            to, the number of tokens included in the split documents will never be above the
                            `max_tokens` value. For example:

                            ```python
                            DocumentSplitter(split_by='sentence', split_length=10, max_tokens=512, max_chars=2000)
                            ```

                            means:

                            - Documents will contain whole sentences
                            - Documents will contain at most 10 sentences
                            - Documents might contain less than 10 sentences if the maximum number of tokens is
                                reached earlier.
                            - Documents will never contain more than 2000 chars. Documents with a content length
                                above that value will be split on the 2000th character.

                            Note that the number of tokens might still be above the maximum if a single sentence
                            contains more than 512 tokens. In this case an `ERROR` log is emitted, but the document
                            is generated with whatever amount of tokens the first sentence has.

                            If the number of units is irrelevant, `split_length` can be safely set at `0`.

        :param tokenizer_model: If `split_by="token"`, you should provide a tokenizer model to compute the tokens,
                                for example `deepset/roberta-base-squad2`. You can give its identifier on Hugging Face Hub,
                                a local path to load it from, or an instance of `PreTrainedTokenizer`.\n
                                If you provide "nltk" instead of a model name, the NLTKWordTokenizer will be used.

        :param nltk_language: If `split_by="sentence"`, the language used by "nltk.tokenize.sent_tokenize", for example "english", or "french".
                                Mind that some languages have limited support by the tokenizer: for example, it seems incapable to split Chinese text
                                by word, but it can correctly split it by sentence.

        :param nltk_folder: If `split_by="sentence"`, specifies the path to the folder containing the NTLK `PunktSentenceTokenizer` models,
                            if loading a model from a local path. Leave empty otherwise.

        :param progress_bar: Whether to show a progress bar.

        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.
        """
        super().__init__()
        self._validate_split_params(
            split_by=split_by,
            split_regex=split_regex,
            split_length=split_length,
            split_overlap=split_overlap,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        self.split_by = split_by
        self.split_regex = split_regex
        self.max_chars = max_chars
        self.max_tokens = max_tokens
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

        self._tokenizer = None
        if tokenizer_model:
            self.tokenizer = tokenizer_model

        if split_by == "token" and not self.tokenizer:
            raise ValueError(
                "If you set split_by='token', you must give a value to 'tokenizer_model'. "
                "Use the same model you're using for your Reader."
            )

    def _validate_split_params(
        self,
        split_by: SplitBy,
        split_regex: Optional[str],
        split_length: int,
        split_overlap: int,
        max_chars: int,
        max_tokens: Optional[int],
    ):
        """
        Performs some basic validation on the parameters of the splitter.
        """
        if split_by not in get_args(SplitBy):
            raise ValueError(f"split_by must be one of: {', '.join(get_args(SplitBy))}")

        if not isinstance(split_length, int) or split_length < 0:
            raise ValueError("split_length must be an integer >= 0")

        if split_length == 0 and not max_tokens:
            logger.warning(
                "split_length is set to 0 and max_tokens is not set. "
                "This means that the documents will be split in chunks of %s chars. "
                "Is this the behavior you expect? If so, set `split_by='character'` "
                "and `split_length=%s` to remove this warning.",
                max_chars,
                max_chars,
            )

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
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer_model=Union[str, Path, PreTrainedTokenizer, TokenizerI]):
        if not tokenizer_model:
            raise ValueError(
                "Can't set the tokenizer to None. "
                "Provide either a Hugging Face identifier, a path to a local tokenizer, "
                "an instance of Haystack's FeatureExtractor, Transformers' PreTrainedTokenizer, "
                "or an NLTK Tokenizer."
            )
        if isinstance(tokenizer_model, (PreTrainedTokenizer, TokenizerI)):
            self._tokenizer = tokenizer_model
        else:
            if tokenizer_model == "nltk":
                self._tokenizer = NLTKWordTokenizer()
            else:
                self._tokenizer = FeatureExtractor(pretrained_model_name_or_path=tokenizer_model)

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
        max_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        add_page_number: Optional[bool] = None,
    ):
        """
        Splits documents into smaller, shorter documents.

        Can split on different units ('character', 'token', 'word', 'sentence', 'paragraph', 'page', or 'regex'),
        at different lengths, and include some overlap across the splits.

        It can also properly assign page numbers and re-assign headlines found in the metadata of the parent document
        to each split document.

        No char is lost in splitting, not even whitespace, and all headlines are preserved, However, text and headlines
        duplication may occur if `split_overlap>0`.

        :param document: The document to split.
        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.
        :param split_regex: If `split_by="regex"`, provide here a regex matching the separator. For example, if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: The maximum number of the above split unit (like word, sentence, page and so on) that are allowed in one document.
                                For instance, if `split_lenght=10` and `split_by="sentence"`, then each output document will contain 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                                For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                                `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                                Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                            length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                            This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.
        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it requires `tokenizer_model` to be set to the
                            model of your Reader and will verify that, whatever your `split_length` value is set
                            to, the number of tokens included in the split documents will never be above the
                            `max_tokens` value. For example:

                            ```python
                            DocumentSplitter(split_by='sentence', split_length=10, max_tokens=512, max_chars=2000)
                            ```

                            means:

                            - Documents will contain whole sentences
                            - Documents will contain at most 10 sentences
                            - Documents might contain less than 10 sentences if the maximum number of tokens is
                                reached earlier.
                            - Documents will never contain more than 2000 chars. Documents with a content length
                                above that value will be split on the 2000th character.

                            Note that the number of tokens might still be above the maximum if a single sentence
                            contains more than 512 tokens. In this case an `ERROR` log is emitted, but the document
                            is generated with whatever amount of tokens the first sentence has.

                            If the number of units is irrelevant, `split_length` can be safely set at 0.

        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.
        """
        split_by = split_by if split_by is not None else self.split_by
        split_regex = split_regex if split_regex is not None else self.split_regex
        split_length = split_length if split_length is not None else self.merger.window_size
        split_overlap = split_overlap if split_overlap is not None else self.merger.window_overlap
        max_chars = max_chars if max_chars is not None else self.max_chars
        add_page_number = add_page_number if add_page_number is not None else self.add_page_number

        self._validate_split_params(
            split_by=split_by,
            split_regex=split_regex,
            split_length=split_length,
            split_overlap=split_overlap,
            max_chars=max_chars,
            max_tokens=max_tokens,
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
            if isinstance(self.tokenizer, TokenizerI):
                splitter_function = lambda text: self.split_by_sentence_tokenizer(text=text)
            else:
                splitter_function = lambda text: self.split_by_dense_tokenizer(text=text)
        else:
            raise ValueError("split_by must be either 'character', 'word', 'sentence', 'paragraph', 'page' or 'regex'")

        final_documents = []
        for document in documents:

            # Split them into single unit documents
            split_documents = self.split_into_units(
                document=document, units=splitter_function(text=document.content), add_page_number=add_page_number
            )[0]

            # If we need to count the tokens, split it by token
            if max_tokens:
                for doc in split_documents:
                    tokens = self.split_by_dense_tokenizer(text=doc.content)[0]
                    doc.meta["tokens_count"] = len(tokens)

            # Merge them back according to the given split_length and split_overlap, if needed
            if (split_length is not None and split_length > 1) or self.merger.window_size > 1:
                split_documents = self.merger.run(
                    documents=split_documents,
                    window_size=split_length,
                    window_overlap=split_overlap,
                    max_tokens=max_tokens,
                )[0]["documents"]

            # If a document longer than max_chars is found, split it into max_length chunks and log loudly.
            sane_documents = []
            for document in split_documents:
                if len(document.content) <= max_chars:
                    sane_documents.append(document)
                else:
                    logger.error(
                        "Found document with a character count higher than the maximum allowed (%s > %s). "
                        "The document is going to be cut at %s chars, so %s chars are being moved to one (or more) new documents. "
                        "Set the maximum amout of characters allowed through the 'max_chars' parameter. "
                        "Keep in mind that very long Documents can severely impact the performance of Readers.",
                        len(document.content),
                        max_chars,
                        max_chars,
                        len(document.content) - max_chars,
                    )
                    hard_splits = [
                        document.content[pos : pos + max_chars] for pos in range(0, len(document.content), max_chars)
                    ]
                    sub_units = hard_splits, [0] * len(hard_splits)
                    sub_documents = self.split_into_units(
                        document=document, units=sub_units, add_page_number=add_page_number
                    )[0]
                    if add_page_number:
                        for sub_document in sub_documents:
                            sub_document.meta["page"] += document.meta["page"] - 1
                    sane_documents += sub_documents

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
        """
        Splits documents into smaller, shorter documents.

        Can split on different units ('character', 'token', 'word', 'sentence', 'paragraph', 'page', or 'regex'),
        at different lengths, and include some overlap across the splits.

        It can also properly assign page numbers and re-assign headlines found in the metadata of the parent document
        to each split document.

        No char is lost in splitting, not even whitespace, and all headlines are preserved, However, text and headlines
        duplication may occur if `split_overlap>0`.

        :param document: The document to split.
        :param split_by: Unit for splitting the document. Can be 'character', 'token', 'word', 'sentence', 'paragraph', 'page', 'regex'.
        :param split_regex: If `split_by="regex"`, provide here a regex matching the separator. For example, if the document
                            should be split on "--my separator--", this field should be `split_regex="--my separator--"`.
        :param split_length: The maximum number of the above split unit (like word, sentence, page and so on) that are allowed in one document.
                                For instance, if `split_lenght=10` and `split_by="sentence"`, then each output document will contain 10 sentences.
        :param split_overlap: Units (for example words or sentences) overlap between two adjacent documents after a split.
                                For example, if `split_by="word" and split_length=5 and split_overlap=2`, then the splits would be like:
                                `[w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]`.
                                Set the value to 0 to ensure there is no overlap among the documents after splitting.
        :param max_chars: Absolute maximum number of chars allowed in a single document. Reaching this boundary
                            cuts the document, even mid-word, and logs a loud error.\n
                            It's recommended to set this value to approximately double the size you expect your documents
                            to be. For example, with `split_by='sentence'`, `split_lenght=2`, if the average sentence
                            length of our document is 100 chars, you should set `max_char=400` or `max_char=500`.\n
                            This is a safety parameter to avoid extremely long documents to end up in the document store.
                            Keep in mind that huge documents (tens of thousands of chars) will strongly impact the
                            performance of Reader nodes and can drastically slow down the indexing speed.
        :param max_tokens:  Maximum number of tokens that are allowed in a single split. If set to 0, it will be
                            ignored. If set to any value above 0, it requires `tokenizer_model` to be set to the
                            model of your Reader and will verify that, whatever your `split_length` value is set
                            to, the number of tokens included in the split documents will never be above the
                            `max_tokens` value. For example:

                            ```python
                            DocumentSplitter(split_by='sentence', split_length=10, max_tokens=512, max_chars=2000)
                            ```

                            means:

                            - Documents will contain whole sentences
                            - Documents will contain at most 10 sentences
                            - Documents might contain less than 10 sentences if the maximum number of tokens is
                                reached earlier.
                            - Documents will never contain more than 2000 chars. Documents with a content length
                                above that value will be split on the 2000th character.

                            Note that the number of tokens might still be above the maximum if a single sentence
                            contains more than 512 tokens. In this case an `ERROR` log is emitted, but the document
                            is generated with whatever amount of tokens the first sentence has.

                            If the number of units is irrelevant, `split_length` can be safely set at 0.

        :param add_page_number: Add the number of the page a paragraph occurs in to the Document's meta
                                field `"page"`. Page boundaries are determined by `"\f"' character which is added
                                in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and
                                `AzureConverter`.
        """
        documents = [
            self.run(
                documents=docs,
                split_by=split_by,
                split_regex=split_regex,
                split_length=split_length,
                split_overlap=split_overlap,
                max_chars=split_max_chars,
                add_page_number=add_page_number,
            )[0]["documents"]
            for docs in tqdm(documents, disable=not self.progress_bar, desc="Splitting", unit="docs")
        ]
        return {"documents": documents}, "output_1"

    def split_into_units(
        self, document: Document, units: Tuple[List[str], List[int]], add_page_number: bool = True
    ) -> Tuple[List[Document], List[int]]:
        """
        Splits the parent document into single unit documents.

        This algorithm is heavily simplified by the lack of split_length and split_overlap. Those are later applied by the merger.

        :param document: The document to split.
        :param units: Two parallel lists of (strings, offsets). Offsets is populated only if the split was performed on a regex
            (so split_by "regex", "page", "paragraph" or "word", but not "sentence", "token" or "character"). In these cases,
            offsets will contain the length of the regex matched to split. Is used by the DocumentCleaner to remove matches
            without having to re-match the documents.
            For example, if split_by="word":
            - units=(['ab ', 'cd. \n\n', 'ef!'], [1, 3, 0]) means that there is one whitespace, three whitespaces and no whitespace
                at the end of each of the strings.
        :param add_page_number: If `True`, ounts the number of form feeds to assign to each split a metadata entry with the page where it starts
                                in the original document.
        """
        if isinstance(document, dict):
            warnings.warn(
                "Use Document objects. Passing a dictionary to DocumentSplitter is deprecated.", DeprecationWarning
            )
            document = Document.from_dict(document)

        if document.content_type != "text":
            raise ValueError(
                f"DocumentSplitter received a '{document.content_type}' document. "
                "Make sure to pass only text documents to it. "
                "You can use a RouteDocuments node to make sure only text document are sent to the DocumentCleaner."
            )
        headlines_to_assign = deepcopy(document.meta.get("headlines", [])) or []
        unit_documents = []
        pages = 1
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
        Splits a long text into chunks based on a regex match.

        :param pattern: The regex to split the text on.
        :param text: The text to split.
        :return: The list of splits, along with the length of the separators matched.
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

        :param text: The text to tokenize.
        :return: The tokenized text as a list of strings.
        """
        token_spans = self.sentence_tokenizer.span_tokenize(text)
        return split_on_spans(text=text, spans=token_spans)

    def split_by_nltk_word_tokenizer(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Splits a given text with an NLTK word tokenizer, preserving all whitespace.

        :param text: The text to tokenize.
        :return: The tokenized text as a list of strings.
        """
        token_spans = self.tokenizer.span_tokenize(text)
        return split_on_spans(text=text, spans=token_spans)

    def split_by_dense_tokenizer(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Splits a given text with a tokenizer, preserving all whitespace.

        :param text: The text to tokenize.
        :return: The tokenized text as a list of strings.
        """
        token_batch = self.tokenizer(text=text)
        token_positions = [pos for pos, i in enumerate(token_batch.sequence_ids()) if i == 0]
        token_chars = [token_batch.token_to_chars(i) for i in token_positions]
        token_spans = [(chars.start, chars.end) for chars in token_chars]
        return split_on_spans(text=text, spans=token_spans)


def split_on_spans(text: str, spans: List[Tuple[int, int]]) -> Tuple[List[str], List[int]]:
    """
    Splits a given text on the arbitrary spans given.

    :param text: The text to tokenize
    :param spans: The spans to split on.
    :return: The tokenized text as a list of strings
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
            logger.exception(
                "Couldn't load sentence tokenizer from %s "
                "Check that the path is correct and that Haystack has permission to access it.",
                tokenizer_model_path,
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "Couldn't find custom sentence tokenizer model for %s in %s. ", language, tokenizer_model_folder
            )
        logger.warning("Trying to find a tokenizer for %s under the default path (tokenizers/punkt/).", language)

    # Try loading from the default path
    if language:
        try:
            return nltk.data.load(f"tokenizers/punkt/{language}.pickle")
        except LookupError as e:
            logger.exception(
                "Couldn't load sentence tokenizer from the default tokenizer path (tokenizers/punkt/). "
                'Make sure NLTk is properly installed, or try running `nltk.download("punkt")` from '
                "any Python shell."
            )
        except (UnpicklingError, ValueError) as e:
            logger.exception(
                "Couldn't find custom sentence tokenizer model for %s in the default tokenizer path (tokenizers/punkt/)"
                'Make sure NLTk is properly installed, or try running `nltk.download("punkt")` from '
                "any Python shell.",
                language,
            )
        logger.warning(
            "Using an English tokenizer as fallback. You may train your own model and use the 'tokenizer_model_folder' parameter."
        )

    # Fallback to English from the default path
    return nltk.data.load("tokenizers/punkt/english.pickle")
