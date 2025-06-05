from copy import deepcopy
from typing import Any, Dict, List, Literal, Tuple

from more_itertools import windowed

from haystack import Document, component, logging
from haystack.components.preprocessors import DocumentSplitter
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install hanlp'") as hanlp_import:
    import hanlp


logger = logging.getLogger(__name__)

# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {"page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}
chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
# Load Chinese sentence slicer
split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)


@component
class ChineseDocumentSplitter(DocumentSplitter):
    def __init__(self, *args, particle_size: Literal["coarse", "fine"] = "coarse", **kwargs):
        """
        A DocumentSplitter for Chinese text.

        'coarse' represents coarse granularity Chinese word segmentation, 'fine' represents fine granularity word
        segmentation, default is coarse granularity word segmentation.

        :param particle_size: The granularity of Chinese word segmentation, either 'coarse' or 'fine'.

        """
        super(ChineseDocumentSplitter, self).__init__(*args, **kwargs)
        self.particle_size = particle_size

        hanlp_import.check()

        self.chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self.chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        self.split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)  # 加载中文的句子切分器

    def _split_by_character(self, doc) -> List[Document]:
        """
        Define a function to handle Chinese clauses

        :param doc:
        :return:
        """
        split_at = _CHARACTER_SPLIT_BY_MAPPING[self.split_by]

        # 'coarse' represents coarse granularity Chinese word segmentation,
        # 'fine' represents fine granularity word segmentation,
        #  default is coarse granularity word segmentation

        if self.language == "zh" and self.particle_size == "coarse":
            units = self.chinese_tokenizer_coarse(doc.content)

        if self.language == "zh" and self.particle_size == "fine":
            units = self.chinese_tokenizer_fine(doc.content)

        if self.language == "en":
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

    # Define a function to handle Chinese clauses
    def chinese_sentence_split(self, text: str) -> list:
        """Split Chinese text into sentences."""
        # Split sentences
        sentences = split_sent(text)

        # Organize the format of segmented sentences
        results = []
        start = 0
        for sentence in sentences:
            start = text.find(sentence, start)
            end = start + len(sentence)
            results.append({"sentence": sentence + "\n", "start": start, "end": end})
            start = end

        return results

    def _split_document(self, doc: Document) -> List[Document]:
        if self.split_by == "sentence" or self.respect_sentence_boundary:
            return self._split_by_nltk_sentence(doc)

        if self.split_by == "function" and self.splitting_function is not None:
            return self._split_by_function(doc)

        return self._split_by_character(doc)

    @staticmethod
    def _concatenate_sentences_based_on_word_amount(
        sentences: List[str], split_length: int, split_overlap: int, language: str, particle_size: str
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
        # chinese_tokenizer_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # chinese_tokenizer_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if language == "zh" and particle_size == "coarse":
                chunk_word_count += len(chinese_tokenizer_coarse(sentence))
                next_sentence_word_count = (
                    len(chinese_tokenizer_coarse(sentences[sentence_idx + 1]))
                    if sentence_idx < len(sentences) - 1
                    else 0
                )
            if language == "zh" and particle_size == "fine":
                chunk_word_count += len(chinese_tokenizer_fine(sentence))
                next_sentence_word_count = (
                    len(chinese_tokenizer_fine(sentences[sentence_idx + 1])) if sentence_idx < len(sentences) - 1 else 0
                )

            # Number of words in the current chunk plus the next sentence is larger than the split_length,
            # or we reached the last sentence
            if (chunk_word_count + next_sentence_word_count) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = ChineseDocumentSplitter._number_of_sentences_to_keep(
                    sentences=current_chunk,
                    split_length=split_length,
                    split_overlap=split_overlap,
                    language=language,
                    particle_size=particle_size,
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

    # Add Chinese sentence segmentation and enable it using language=="zh"
    def _split_by_nltk_sentence(self, doc: Document) -> List[Document]:
        split_docs = []

        if self.language == "zh":
            result = self.chinese_sentence_split(doc.content)
        if self.language == "en":
            result = self.sentence_splitter.split_sentences(doc.content)  # type: ignore # None check is done in run()

        units = [sentence["sentence"] for sentence in result]

        if self.respect_sentence_boundary:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_sentences_based_on_word_amount(
                sentences=units,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                language=self.language,
                particle_size=self.particle_size,
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
            copied_meta = deepcopy(meta)
            copied_meta["page_number"] = splits_pages[i]
            copied_meta["split_id"] = i
            copied_meta["split_idx_start"] = split_idx
            doc = Document(content=txt, meta=copied_meta)
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

        for d in documents:
            d.content = d.content.replace(" ", "")
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
            # type: ignore
            overlapping_str = previous_doc.content[overlapping_range[0] : overlapping_range[1]]

            if current_doc.content.startswith(overlapping_str):  # type: ignore
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append({"doc_id": previous_doc.id, "range": overlapping_range})

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append({"doc_id": current_doc.id, "range": overlapping_range})

    @staticmethod
    def _number_of_sentences_to_keep(
        sentences: List[str], split_length: int, split_overlap: int, language: str, particle_size: str
    ) -> int:
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

        for sent in reversed(sentences[1:]):
            if language == "zh" and particle_size == "coarse":
                num_words += len(chinese_tokenizer_coarse(sent))
            if language == "zh" and particle_size == "fine":
                num_words += len(chinese_tokenizer_fine(sent))
            # If the number of words is larger than the split_length then don't add any more sentences
            if num_words > split_length:
                break
            num_sentences_to_keep += 1
            if num_words > split_overlap:
                break
        return num_sentences_to_keep
