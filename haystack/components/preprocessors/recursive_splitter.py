# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, List, Optional

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class RecursiveDocumentSplitter:
    """
    Recursively chunk text into smaller chunks.

    This component is used to split text into smaller chunks, it does so by recursively applying a list of separators
    to the text.

    Each separator is applied to the text, if then checks each of the resulting chunks, it keeps the ones chunks that
    are within the chunk_size, for the ones that are larger than the chunk_size, it applies the next separator in the
    list to the remaining text.

    This is done until all chunks are smaller than the chunk_size parameter.

    Example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=260, chunk_overlap=0, separators=["\n\n", "\n", ".", " "], keep_separator=True)
    text = '''Artificial intelligence (AI) - Introduction

    AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
    AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games.'''

    doc = Document(content=text)
    doc_chunks = chunker.run([doc])
    >[
    >Document(id=..., content: 'Artificial intelligence (AI) - Introduction\n\n', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951'}),
    >Document(id=..., content: 'AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems. ', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951'}),
    >Document(id=..., content: 'AI technology is widely used throughout industry, government, and science.', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951'}),
    >Document(id=..., content: ' Some high-profile applications include advanced web search engines; recommendation systems; interac...', meta: {'original_id': '65167a9823dd883de577e828ca4fd529e6f7241f0ff616acfce454d808478951'})
    >]
    """  # noqa: E501

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        split_length: int = 200,
        split_overlap: int = 0,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """
        Initializes a RecursiveDocumentSplitter.

        :param split_length: The maximum length of each chunk.
        :param split_overlap: The number of characters to overlap between consecutive chunks.
        :param separators: A list of separator characters to use for splitting the text. If the separator is "sentence",
                the text will be split into sentences using a custom sentence tokenizer based on NLTK.
        :param keep_separator: Whether to keep the separator character in the resulting chunks.
        :param is_separator_regex: Whether the separator is a regular expression.

        :raises ValueError: If the overlap is greater than or equal to the chunk size or if the overlap is negative, or
                            if any separator is not a string.
        """
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.separators = separators if separators else ["\n\n", "\n", ".", " "]
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self._check_params()
        if separators and "sentence" in separators:
            self.nltk_tokenizer = self._get_custom_sentence_tokenizer()

    def _check_params(self):
        if self.split_overlap < 0:
            raise ValueError("Overlap must be greater than zero.")
        if self.split_overlap >= self.split_length:
            raise ValueError("Overlap cannot be greater than or equal to the chunk size.")
        if not all(isinstance(separator, str) for separator in self.separators):
            raise ValueError("All separators must be strings.")

    @staticmethod
    def _get_custom_sentence_tokenizer():
        try:
            from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter
        except (LookupError, ModuleNotFoundError):
            raise Exception("You need to install NLTK to use this function. You can install it via `pip install nltk`")
        return SentenceSplitter()

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
            current_chunk = chunks[idx - 1][overlap_start:] + chunk
            overlapped_chunks.append(current_chunk)
        return overlapped_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Recursive chunking algorithm that divides text into smaller chunks based on a list of separator characters.

        It starts with a list of separator characters (e.g., ["\n\n", "\n", " ", ""]) and attempts to divide the text
        using the first separator. If the resulting chunks are still larger than the specified chunk size, it moves to
        the next separator in the list. This process continues recursively, progressively applying each specific
        separator until the chunks meet the desired size criteria.

        :param text: The text to be split into chunks.
        :returns:
            A list of text chunks.
        """
        if len(text) <= self.split_length:
            return [text]

        # type ignore below because we already checked that separators is not None
        # try each separator
        for separator in self.separators:  # type: ignore
            if separator in "sentence":  # using nltk sentence tokenizer
                sentence_with_spans = self.nltk_tokenizer.split_sentences(text)
                splits = [sentence["sentence"] for sentence in sentence_with_spans]
            else:
                # split using the current separator
                splits = text.split(separator) if not self.is_separator_regex else re.split(separator, text)

            # filter out empty splits
            splits = [s for s in splits if s.strip()]

            if len(splits) == 1:  # go to next separator, if current separator not found
                continue

            chunks = []
            current_chunk: List[str] = []
            current_length = 0

            # check splits, if any is too long, recursively chunk it, otherwise add to current chunk
            for split in splits:
                split_text = split
                if self.keep_separator and separator != "sentence":
                    split_text = split + separator

                # if adding this split exceeds chunk_size, process current_chunk
                if current_length + len(split_text) > self.split_length:
                    if current_chunk:  # keep the good splits
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # recursively handle splits that are too large
                    if len(split_text) > self.split_length:
                        chunks.extend(self._chunk_text(split_text))
                    else:
                        chunks.append(split_text)
                else:
                    current_chunk.append(split_text)
                    current_length += len(split_text)

            if current_chunk:
                chunks.append("".join(current_chunk))

            if self.split_overlap > 0:
                chunks = self._apply_overlap(chunks)

            return chunks

        # if no separator worked, fall back to character-level chunking
        return [text[i : i + self.split_length] for i in range(0, len(text), self.split_length - self.split_overlap)]

    def _run_one(self, doc: Document) -> List[Document]:
        new_docs = []
        # NOTE: the check for a non-empty content is already done in the run method, hence the type ignore below
        chunks = self._chunk_text(doc.content)  # type: ignore
        for chunk in chunks:
            new_doc = Document(content=chunk, meta=doc.meta)
            new_doc.meta["original_id"] = doc.id
            new_docs.append(new_doc)
        return new_docs

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split documents into Documents with smaller chunks of text.

        :param documents: List of Documents to split.
        :returns:
            A dictionary containing a key "documents" with a List of Documents with smaller chunks of text corresponding
            to the input documents.
        """
        new_docs = []
        for doc in documents:
            if not doc.content or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            new_docs.extend(self._run_one(doc))
        return {"documents": new_docs}
