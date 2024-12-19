# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from haystack import Document, component, logging
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from haystack.core.serialization import default_to_dict
from haystack.utils import serialize_callable

logger = logging.getLogger(__name__)


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

        warnings.warn(
            "The NLTKDocumentSplitter is deprecated and will be removed in the next release. "
            "See DocumentSplitter which now supports the functionalities of the NLTKDocumentSplitter, i.e.: "
            "using NLTK to detect sentence boundaries.",
            DeprecationWarning,
        )

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
        self.sentence_splitter = None
        self.language = language

    def warm_up(self):
        """
        Warm up the NLTKDocumentSplitter by loading the sentence tokenizer.
        """
        if self.sentence_splitter is None:
            self.sentence_splitter = SentenceSplitter(
                language=self.language,
                use_split_rules=self.use_split_rules,
                extend_abbreviations=self.extend_abbreviations,
                keep_white_spaces=True,
            )

    def _split_into_units(
        self, text: str, split_by: Literal["function", "page", "passage", "period", "sentence", "word", "line"]
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
            assert self.sentence_splitter is not None
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
        if self.sentence_splitter is None:
            raise RuntimeError(
                "The component NLTKDocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

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

    @staticmethod
    def _concatenate_sentences_based_on_word_amount(
        sentences: List[str], split_length: int, split_overlap: int
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
                num_sentences_to_keep = NLTKDocumentSplitter._number_of_sentences_to_keep(
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
