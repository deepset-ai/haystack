# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class RecursiveDocumentSplitter:
    """
    Recursively chunk text into smaller chunks.

    This component is used to split text into smaller chunks, it does so by recursively applying a list of separators
    to the text.

    The separators are applied in the order they are provided, typically this is a list of separators that are
    applied in a specific order, being the last separator the most specific one.

    Each separator is applied to the text, it then checks each of the resulting chunks, it keeps the chunks that
    are within the chunk_size, for the ones that are larger than the chunk_size, it applies the next separator in the
    list to the remaining text.

    This is done until all chunks are smaller than the chunk_size parameter.

    Example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import RecursiveDocumentSplitter

    chunker = RecursiveDocumentSplitter(split_length=260, split_overlap=0, separators=["\n\n", "\n", ".", " "])
    text = '''Artificial intelligence (AI) - Introduction

    AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
    AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games.'''

    doc = Document(content=text)
    doc_chunks = chunker.run([doc])
    print(doc_chunks["documents"])
    >[
    >Document(id=..., content: 'Artificial intelligence (AI) - Introduction\n\n', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951', 'split_id': 0, 'split_idx_start': 0, '_split_overlap': []})
    >Document(id=..., content: 'AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951', 'split_id': 1, 'split_idx_start': 45, '_split_overlap': []})
    >Document(id=..., content: 'AI technology is widely used throughout industry, government, and science.', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951', 'split_id': 2, 'split_idx_start': 142, '_split_overlap': []})
    >Document(id=..., content: ' Some high-profile applications include advanced web search engines; recommendation systems; interac...', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951', 'split_id': 3, 'split_idx_start': 216, '_split_overlap': []})
    >]
    """  # noqa: E501

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        split_length: int = 200,
        split_overlap: int = 0,
        split_units: Literal["words", "char"] = "char",
        separators: Optional[List[str]] = None,
        sentence_splitter_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a RecursiveDocumentSplitter.

        :param split_length: The maximum length of each chunk in characters.
        :param split_overlap: The number of characters to overlap between consecutive chunks.
        :param split_units: The unit of the split_length parameter. It can be either "words" or "char".
        :param separators: An optional list of separator strings to use for splitting the text. The string
            separators will be treated as regular expressions unless the separator is "sentence", in that case the
            text will be split into sentences using a custom sentence tokenizer based on NLTK.
            See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter.
            If no separators are provided, the default separators ["\n\n", "sentence", "\n", " "] are used.

        :raises ValueError: If the overlap is greater than or equal to the chunk size or if the overlap is negative, or
                            if any separator is not a string.
        """
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_units = split_units
        self.separators = separators if separators else ["\n\n", "sentence", "\n", " "]  # default separators
        self.sentence_tokenizer_params = sentence_splitter_params
        self._check_params()
        if "sentence" in self.separators:
            sentence_splitter_params = sentence_splitter_params or {"keep_white_spaces": True}
            self.nltk_tokenizer = self._get_custom_sentence_tokenizer(sentence_splitter_params)

    def _check_params(self):
        if self.split_length < 1:
            raise ValueError("Split length must be at least 1 character.")
        if self.split_overlap < 0:
            raise ValueError("Overlap must be greater than zero.")
        if self.split_overlap >= self.split_length:
            raise ValueError("Overlap cannot be greater than or equal to the chunk size.")
        if not all(isinstance(separator, str) for separator in self.separators):
            raise ValueError("All separators must be strings.")

    @staticmethod
    def _get_custom_sentence_tokenizer(sentence_splitter_params: Dict[str, Any]):
        from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter

        return SentenceSplitter(**sentence_splitter_params)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Applies an overlap between consecutive chunks if the chunk_overlap attribute is greater than zero.

        :param chunks: List of text chunks.
        :returns:
            The list of chunks with overlap applied.
        """
        overlapped_chunks = []

        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
                continue
            overlap_start = max(0, len(chunks[idx - 1]) - self.split_overlap)
            overlap = chunks[idx - 1][overlap_start:]
            if overlap == chunks[idx - 1]:
                logger.warning(
                    "Overlap is the same as the previous chunk. "
                    "Consider increasing the `split_overlap` parameter or decreasing the `split_length` parameter."
                )
            current_chunk = overlap + chunk
            overlapped_chunks.append(current_chunk)

        return overlapped_chunks

    def _chunk_length(self, text: str) -> int:
        """
        Get the length of the chunk in words or characters.

        :param text: The text to be split into chunks.
        :returns:
            The length of the chunk in words or characters.
        """
        if self.split_units == "words":
            return len(text.split())
        else:
            return len(text)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Recursive chunking algorithm that divides text into smaller chunks based on a list of separator characters.

        It starts with a list of separator characters (e.g., ["\n\n", "sentence", "\n", " "]) and attempts to divide
        the text using the first separator. If the resulting chunks are still larger than the specified chunk size,
        it moves to the next separator in the list. This process continues recursively, progressively applying each
        specific separator until the chunks meet the desired size criteria.

        :param text: The text to be split into chunks.
        :returns:
            A list of text chunks.
        """
        if self._chunk_length(text) <= self.split_length:
            return [text]

        for curr_separator in self.separators:  # type: ignore # the caller already checked that separators is not None
            if curr_separator == "sentence":
                sentence_with_spans = self.nltk_tokenizer.split_sentences(text)
                splits = [sentence["sentence"] for sentence in sentence_with_spans]
            else:
                escaped_separator = re.escape(curr_separator)
                escaped_separator = (
                    f"({escaped_separator})"  # wrap the separator in a group to include it in the splits
                )
                splits = re.split(escaped_separator, text)

                # merge every two consecutive splits, i.e.: the text and the separator after it
                splits = [
                    "".join([splits[i], splits[i + 1]]) if i < len(splits) - 1 else splits[i]
                    for i in range(0, len(splits), 2)
                ]

                # remove last split if it is empty
                splits = splits[:-1] if splits[-1] == "" else splits

            if len(splits) == 1:  # go to next separator, if current separator not found in the text
                continue

            chunks = []
            current_chunk: List[str] = []
            current_length = 0

            # check splits, if any is too long, recursively chunk it, otherwise add to current chunk
            for split in splits:
                split_text = split
                # if adding this split exceeds chunk_size, process current_chunk
                if current_length + self._chunk_length(split_text) > self.split_length:
                    if current_chunk:  # keep the good splits
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # recursively handle splits that are too large
                    if self._chunk_length(split_text) > self.split_length:
                        if curr_separator == self.separators[-1]:
                            # tried the last separator, can't split further, break the loop and fall back to
                            # character-level chunking
                            break
                        chunks.extend(self._chunk_text(split_text))
                    else:
                        chunks.append(split_text)
                else:
                    current_chunk.append(split_text)
                    current_length += self._chunk_length(split_text)

            if current_chunk:
                chunks.append("".join(current_chunk))

            if self.split_overlap > 0:
                chunks = self._apply_overlap(chunks)

            if chunks:
                return chunks

        # if no separator worked, fall back to character-level chunking
        return [
            text[i : i + self.split_length]
            for i in range(0, self._chunk_length(text), self.split_length - self.split_overlap)
        ]

    def _run_one(self, doc: Document) -> List[Document]:
        new_docs: List[Document] = []
        chunks = self._chunk_text(doc.content)  # type: ignore # the caller already check for a non-empty doc.content
        chunks = chunks[:-1] if len(chunks[-1]) == 0 else chunks  # remove last empty chunk
        current_position = 0
        current_page = 1

        for split_nr, chunk in enumerate(chunks):
            new_doc = Document(content=chunk, meta=deepcopy(doc.meta))
            new_doc.meta["split_id"] = split_nr
            new_doc.meta["split_idx_start"] = current_position
            new_doc.meta["_split_overlap"] = [] if self.split_overlap > 0 else None

            if split_nr > 0 and self.split_overlap > 0:
                previous_doc = new_docs[-1]
                overlap_length = len(previous_doc.content) - (current_position - previous_doc.meta["split_idx_start"])  # type: ignore
                if overlap_length > 0:
                    previous_doc.meta["_split_overlap"].append({"doc_id": new_doc.id, "range": (0, overlap_length)})
                    new_doc.meta["_split_overlap"].append(
                        {
                            "doc_id": previous_doc.id,
                            "range": (len(previous_doc.content) - overlap_length, len(previous_doc.content)),  # type: ignore
                        }
                    )

            # count page breaks in the chunk
            current_page += chunk.count("\f")

            # if there are consecutive page breaks at the end with no more text, adjust the page number
            # e.g: "text\f\f\f" -> 3 page breaks, but current_page should be 1
            consecutive_page_breaks = len(chunk) - len(chunk.rstrip("\f"))

            if consecutive_page_breaks > 0:
                new_doc.meta["page_number"] = current_page - consecutive_page_breaks
            else:
                new_doc.meta["page_number"] = current_page

            # keep the new chunk doc and update the current position
            new_docs.append(new_doc)
            current_position += len(chunk) - (self.split_overlap if split_nr < len(chunks) - 1 else 0)

        return new_docs

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split a list of documents into documents with smaller chunks of text.

        :param documents: List of Documents to split.
        :returns:
            A dictionary containing a key "documents" with a List of Documents with smaller chunks of text corresponding
            to the input documents.
        """
        docs = []
        for doc in documents:
            if not doc.content or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            docs.extend(self._run_one(doc))

        return {"documents": docs}
