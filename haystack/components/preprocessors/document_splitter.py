# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from more_itertools import windowed

from haystack import Document, component, logging
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)

# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {"page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}


@component
class DocumentSplitter:
    """
    Splits long documents into smaller chunks.

    This is a common preprocessing step during indexing. It helps Embedders create meaningful semantic representations
    and prevents exceeding language model context limits.

    The DocumentSplitter is compatible with the following DocumentStores:
    - [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
    - [Chroma](https://docs.haystack.deepset.ai/docs/chromadocumentstore) limited support, overlapping information is
      not stored
    - [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
    - [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
    - [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
    - [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store) limited support, overlapping
       information is not stored
    - [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)
    - [Weaviate](https://docs.haystack.deepset.ai/docs/weaviatedocumentstore)

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import DocumentSplitter

    doc = Document(content="Moonlight shimmered softly, wolves howled nearby, night enveloped everything.")

    splitter = DocumentSplitter(split_by="word", split_length=3, split_overlap=0)
    result = splitter.run(documents=[doc])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        split_by: Literal["function", "page", "passage", "period", "word", "line", "sentence"] = "word",
        split_length: int = 200,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = False,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
    ):
        """
        Initialize DocumentSplitter.

        :param split_by: The unit for splitting your documents. Choose from:
            - `word` for splitting by spaces (" ")
            - `period` for splitting by periods (".")
            - `page` for splitting by form feed ("\\f")
            - `passage` for splitting by double line breaks ("\\n\\n")
            - `line` for splitting each line ("\\n")
            - `sentence` for splitting by NLTK sentence tokenizer

        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of overlapping units for each split.
        :param split_threshold: The minimum number of units per split. If a split has fewer units
            than the threshold, it's attached to the previous split.
        :param splitting_function: Necessary when `split_by` is set to "function".
            This is a function which must accept a single `str` as input and return a `list` of `str` as output,
            representing the chunks after splitting.
        :param respect_sentence_boundary: Choose whether to respect sentence boundaries when splitting by "word".
            If True, uses NLTK to detect sentence boundaries, ensuring splits occur only between sentences.
        :param language: Choose the language for the NLTK tokenizer. The default is English ("en").
        :param use_split_rules: Choose whether to use additional split rules when splitting by `sentence`.
        :param extend_abbreviations: Choose whether to extend NLTK's PunktTokenizer abbreviations with a list
            of curated abbreviations, if available. This is currently supported for English ("en") and German ("de").
        """

        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function
        self.respect_sentence_boundary = respect_sentence_boundary
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations

        self._init_checks(
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
            splitting_function=splitting_function,
            respect_sentence_boundary=respect_sentence_boundary,
        )
        self._use_sentence_splitter = split_by == "sentence" or (respect_sentence_boundary and split_by == "word")
        if self._use_sentence_splitter:
            nltk_imports.check()
            self.sentence_splitter = None

        if split_by == "sentence":
            # ToDo: remove this warning in the next major release
            msg = (
                "The `split_by='sentence'` no longer splits by '.' and now relies on custom sentence tokenizer "
                "based on NLTK. To achieve the previous behaviour use `split_by='period'."
            )
            warnings.warn(msg)

    def _init_checks(
        self,
        *,
        split_by: str,
        split_length: int,
        split_overlap: int,
        splitting_function: Optional[Callable],
        respect_sentence_boundary: bool,
    ) -> None:
        """
        Validates initialization parameters for DocumentSplitter.

        :param split_by: The unit for splitting documents
        :param split_length: The maximum number of units in each split
        :param split_overlap: The number of overlapping units for each split
        :param splitting_function: Custom function for splitting when split_by="function"
        :param respect_sentence_boundary: Whether to respect sentence boundaries when splitting
        :raises ValueError: If any parameter is invalid
        """
        valid_split_by = ["function", "page", "passage", "period", "word", "line", "sentence"]
        if split_by not in valid_split_by:
            raise ValueError(f"split_by must be one of {', '.join(valid_split_by)}.")

        if split_by == "function" and splitting_function is None:
            raise ValueError("When 'split_by' is set to 'function', a valid 'splitting_function' must be provided.")

        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")

        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")

        if respect_sentence_boundary and split_by != "word":
            logger.warning(
                "The 'respect_sentence_boundary' option is only supported for `split_by='word'`. "
                "The option `respect_sentence_boundary` will be set to `False`."
            )
            self.respect_sentence_boundary = False

    def warm_up(self):
        """
        Warm up the DocumentSplitter by loading the sentence tokenizer.
        """
        if self._use_sentence_splitter and self.sentence_splitter is None:
            self.sentence_splitter = SentenceSplitter(
                language=self.language,
                use_split_rules=self.use_split_rules,
                extend_abbreviations=self.extend_abbreviations,
                keep_white_spaces=True,
            )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Split documents into smaller parts.

        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.
        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `page_number` to track the original page number.
                - All other metadata copied from the original document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """
        if self._use_sentence_splitter and self.sentence_splitter is None:
            raise RuntimeError(
                "The component DocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs: List[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but content for document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            split_docs += self._split_document(doc)
        return {"documents": split_docs}

    def _split_document(self, doc: Document) -> List[Document]:
        if self.split_by == "sentence" or self.respect_sentence_boundary:
            return self._split_by_nltk_sentence(doc)

        if self.split_by == "function" and self.splitting_function is not None:
            return self._split_by_function(doc)

        return self._split_by_character(doc)

    def _split_by_nltk_sentence(self, doc: Document) -> List[Document]:
        split_docs = []

        result = self.sentence_splitter.split_sentences(doc.content)  # type: ignore # None check is done in run()
        units = [sentence["sentence"] for sentence in result]

        if self.respect_sentence_boundary:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_sentences_based_on_word_amount(
                sentences=units, split_length=self.split_length, split_overlap=self.split_overlap
            )
        else:
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

        return split_docs

    def _split_by_character(self, doc) -> List[Document]:
        split_at = _CHARACTER_SPLIT_BY_MAPPING[self.split_by]
        units = doc.content.split(split_at)
        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += split_at
        text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
            units, self.split_length, self.split_overlap, self.split_threshold
        )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        return self._create_docs_from_splits(
            text_splits=text_splits, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

    def _split_by_function(self, doc) -> List[Document]:
        # the check for None is done already in the run method
        splits = self.splitting_function(doc.content)  # type: ignore
        docs: List[Document] = []
        for s in splits:
            meta = deepcopy(doc.meta)
            meta["source_id"] = doc.id
            docs.append(Document(content=s, meta=meta))
        return docs

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_threshold: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Concatenates the elements into parts of split_length units.

        Keeps track of the original page number that each element belongs. If the length of the current units is less
        than the pre-defined `split_threshold`, it does not create a new split. Instead, it concatenates the current
        units with the last split, preventing the creation of excessively small splits.
        """

        text_splits: List[str] = []
        splits_pages: List[int] = []
        splits_start_idxs: List[int] = []
        cur_start_idx = 0
        cur_page = 1
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)

        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)

            # check if length of current units is below split_threshold
            if len(current_units) < split_threshold and len(text_splits) > 0:
                # concatenate the last split with the current one
                text_splits[-1] += txt

            # NOTE: This line skips documents that have content=""
            elif len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)

            processed_units = current_units[: split_length - split_overlap]
            cur_start_idx += len("".join(processed_units))

            if self.split_by == "page":
                num_page_breaks = len(processed_units)
            else:
                num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)

            cur_page += num_page_breaks

        return text_splits, splits_pages, splits_start_idxs

    def _create_docs_from_splits(
        self, text_splits: List[str], splits_pages: List[int], splits_start_idxs: List[int], meta: Dict[str, Any]
    ) -> List[Document]:
        """
        Creates Document objects from splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, (txt, split_idx) in enumerate(zip(text_splits, splits_start_idxs)):
            meta = deepcopy(meta)
            doc = Document(content=txt, meta=meta)
            doc.meta["page_number"] = splits_pages[i]
            doc.meta["split_id"] = i
            doc.meta["split_idx_start"] = split_idx
            documents.append(doc)

            if self.split_overlap <= 0:
                continue

            doc.meta["_split_overlap"] = []

            if i == 0:
                continue

            doc_start_idx = splits_start_idxs[i]
            previous_doc = documents[i - 1]
            previous_doc_start_idx = splits_start_idxs[i - 1]
            self._add_split_overlap_information(doc, doc_start_idx, previous_doc, previous_doc_start_idx)

        return documents

    @staticmethod
    def _add_split_overlap_information(
        current_doc: Document, current_doc_start_idx: int, previous_doc: Document, previous_doc_start_idx: int
    ):
        """
        Adds split overlap information to the current and previous Document's meta.

        :param current_doc: The Document that is being split.
        :param current_doc_start_idx: The starting index of the current Document.
        :param previous_doc: The Document that was split before the current Document.
        :param previous_doc_start_idx: The starting index of the previous Document.
        """
        overlapping_range = (current_doc_start_idx - previous_doc_start_idx, len(previous_doc.content))  # type: ignore

        if overlapping_range[0] < overlapping_range[1]:
            overlapping_str = previous_doc.content[overlapping_range[0] : overlapping_range[1]]  # type: ignore

            if current_doc.content.startswith(overlapping_str):  # type: ignore
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append({"doc_id": previous_doc.id, "range": overlapping_range})

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append({"doc_id": current_doc.id, "range": overlapping_range})

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSplitter":
        """
        Deserializes the component from a dictionary.
        """
        init_params = data.get("init_parameters", {})

        splitting_function = init_params.get("splitting_function", None)
        if splitting_function:
            init_params["splitting_function"] = deserialize_callable(splitting_function)

        return default_from_dict(cls, data)

    @staticmethod
    def _concatenate_sentences_based_on_word_amount(
        sentences: List[str], split_length: int, split_overlap: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Groups the sentences into chunks of `split_length` words while respecting sentence boundaries.

        This function is only used when splitting by `word` and `respect_sentence_boundary` is set to `True`, i.e.:
        with NLTK sentence tokenizer.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: A tuple containing the concatenated sentences, the start page numbers, and the start indices.
        """
        # chunk information
        chunk_word_count = 0
        chunk_starting_page_number = 1
        chunk_start_idx = 0
        current_chunk: List[str] = []
        # output lists
        split_start_page_numbers = []
        list_of_splits: List[List[str]] = []
        split_start_indices = []

        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            chunk_word_count += len(sentence.split())
            next_sentence_word_count = (
                len(sentences[sentence_idx + 1].split()) if sentence_idx < len(sentences) - 1 else 0
            )

            # Number of words in the current chunk plus the next sentence is larger than the split_length,
            # or we reached the last sentence
            if (chunk_word_count + next_sentence_word_count) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = DocumentSplitter._number_of_sentences_to_keep(
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
