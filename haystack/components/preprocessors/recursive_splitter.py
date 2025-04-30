# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

from haystack import Document, component, logging
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install tiktoken'") as tiktoken_imports:
    import tiktoken

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
    are within the split_length, for the ones that are larger than the split_length, it applies the next separator in the
    list to the remaining text.

    This is done until all chunks are smaller than the split_length parameter.

    Example:

    ```python
    from haystack import Document
    from haystack.components.preprocessors import RecursiveDocumentSplitter

    chunker = RecursiveDocumentSplitter(split_length=260, split_overlap=0, separators=["\\n\\n", "\\n", ".", " "])
    text = ('''Artificial intelligence (AI) - Introduction

    AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
    AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines; recommendation systems; interacting via human speech; autonomous vehicles; generative and creative tools; and superhuman play and analysis in strategy games.''')
    chunker.warm_up()
    doc = Document(content=text)
    doc_chunks = chunker.run([doc])
    print(doc_chunks["documents"])
    >[
    >Document(id=..., content: 'Artificial intelligence (AI) - Introduction\\n\\n', meta: {'original_id': '...', 'split_id': 0, 'split_idx_start': 0, '_split_overlap': []})
    >Document(id=..., content: 'AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\\n', meta: {'original_id': '...', 'split_id': 1, 'split_idx_start': 45, '_split_overlap': []})
    >Document(id=..., content: 'AI technology is widely used throughout industry, government, and science.', meta: {'original_id': '...', 'split_id': 2, 'split_idx_start': 142, '_split_overlap': []})
    >Document(id=..., content: ' Some high-profile applications include advanced web search engines; recommendation systems; interac...', meta: {'original_id': '...', 'split_id': 3, 'split_idx_start': 216, '_split_overlap': []})
    >]
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        split_length: int = 200,
        split_overlap: int = 0,
        split_unit: Literal["word", "char", "token"] = "word",
        separators: Optional[List[str]] = None,
        sentence_splitter_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a RecursiveDocumentSplitter.

        :param split_length: The maximum length of each chunk by default in words, but can be in characters or tokens.
            See the `split_units` parameter.
        :param split_overlap: The number of characters to overlap between consecutive chunks.
        :param split_unit: The unit of the split_length parameter. It can be either "word", "char", or "token".
            If "token" is selected, the text will be split into tokens using the tiktoken tokenizer (o200k_base).
        :param separators: An optional list of separator strings to use for splitting the text. The string
            separators will be treated as regular expressions unless the separator is "sentence", in that case the
            text will be split into sentences using a custom sentence tokenizer based on NLTK.
            See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter.
            If no separators are provided, the default separators ["\\n\\n", "sentence", "\\n", " "] are used.
        :param sentence_splitter_params: Optional parameters to pass to the sentence tokenizer.
            See: haystack.components.preprocessors.sentence_tokenizer.SentenceSplitter for more information.

        :raises ValueError: If the overlap is greater than or equal to the chunk size or if the overlap is negative, or
                            if any separator is not a string.
        """
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_units = split_unit
        self.separators = separators if separators else ["\n\n", "sentence", "\n", " "]  # default separators
        self._check_params()
        self.nltk_tokenizer = None
        self.sentence_splitter_params = (
            {"keep_white_spaces": True} if sentence_splitter_params is None else sentence_splitter_params
        )
        self.tiktoken_tokenizer: Optional["tiktoken.Encoding"] = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the sentence tokenizer and tiktoken tokenizer if needed.
        """
        if "sentence" in self.separators:
            self.nltk_tokenizer = self._get_custom_sentence_tokenizer(self.sentence_splitter_params)
        if self.split_units == "token":
            tiktoken_imports.check()
            self.tiktoken_tokenizer = tiktoken.get_encoding("o200k_base")
        self._is_warmed_up = True

    def _check_params(self) -> None:
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

    def _split_chunk(self, current_chunk: str) -> Tuple[str, str]:
        """
        Splits a chunk based on the split_length and split_units attribute.

        :param current_chunk: The current chunk to be split.
        :returns:
            A tuple containing the current chunk and the remaining chunk.
        """
        if self.split_units == "word":
            words = current_chunk.split()
            current_chunk = " ".join(words[: self.split_length])
            remaining_words = words[self.split_length :]
            return current_chunk, " ".join(remaining_words)
        elif self.split_units == "char":
            text = current_chunk
            current_chunk = text[: self.split_length]
            remaining_chars = text[self.split_length :]
            return current_chunk, remaining_chars
        else:  # token
            # at this point we know that the tokenizer is already initialized
            tokens = self.tiktoken_tokenizer.encode(current_chunk)  # type: ignore
            current_tokens = tokens[: self.split_length]
            remaining_tokens = tokens[self.split_length :]
            return self.tiktoken_tokenizer.decode(current_tokens), self.tiktoken_tokenizer.decode(remaining_tokens)  # type: ignore

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Applies an overlap between consecutive chunks if the chunk_overlap attribute is greater than zero.

        Works for both word- and character-level splitting. It trims the last chunk if it exceeds the split_length and
        adds the trimmed content to the next chunk. If the last chunk is still too long after trimming, it splits it
        and adds the first chunk to the list. This process continues until the last chunk is within the split_length.

        :param chunks: A list of text chunks.
        :returns:
            A list of text chunks with the overlap applied.
        """
        overlapped_chunks: List[str] = []

        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
                continue

            # get the overlap between the current and previous chunk
            overlap, prev_chunk = self._get_overlap(overlapped_chunks)
            if overlap == prev_chunk:
                logger.warning(
                    "Overlap is the same as the previous chunk. "
                    "Consider increasing the `split_length` parameter or decreasing the `split_overlap` parameter."
                )

            current_chunk = self._create_chunk_starting_with_overlap(chunk, overlap)

            # if this new chunk exceeds 'split_length', trim it and move the remaining text to the next chunk
            # if this is the last chunk, another new chunk will contain the trimmed text preceded by the overlap
            # of the last chunk
            if self._chunk_length(current_chunk) > self.split_length:
                current_chunk, remaining_text = self._split_chunk(current_chunk)
                if idx < len(chunks) - 1:
                    if self.split_units == "word":
                        chunks[idx + 1] = remaining_text + " " + chunks[idx + 1]
                    elif self.split_units == "token":
                        # For token-based splitting, combine at token level
                        # at this point we know that the tokenizer is already initialized
                        remaining_tokens = self.tiktoken_tokenizer.encode(remaining_text)  # type: ignore
                        next_chunk_tokens = self.tiktoken_tokenizer.encode(chunks[idx + 1])  # type: ignore
                        chunks[idx + 1] = self.tiktoken_tokenizer.decode(remaining_tokens + next_chunk_tokens)  # type: ignore
                    else:  # char
                        chunks[idx + 1] = remaining_text + chunks[idx + 1]
                elif remaining_text:
                    # create a new chunk with the trimmed text preceded by the overlap of the last chunk
                    overlapped_chunks.append(current_chunk)
                    chunk = remaining_text
                    overlap, _ = self._get_overlap(overlapped_chunks)
                    current_chunk = self._create_chunk_starting_with_overlap(chunk, overlap)

            overlapped_chunks.append(current_chunk)

            # it can still be that the new last chunk exceeds the 'split_length'
            # continue splitting until the last chunk is within 'split_length'
            if idx == len(chunks) - 1 and self._chunk_length(current_chunk) > self.split_length:
                last_chunk = overlapped_chunks.pop()
                first_chunk, remaining_chunk = self._split_chunk(last_chunk)
                overlapped_chunks.append(first_chunk)

                while remaining_chunk:
                    # combine overlap with remaining chunk
                    overlap, _ = self._get_overlap(overlapped_chunks)
                    current = self._create_chunk_starting_with_overlap(remaining_chunk, overlap)

                    # if it fits within split_length we are done
                    if self._chunk_length(current) <= self.split_length:
                        overlapped_chunks.append(current)
                        break

                    # otherwise split it again
                    first_chunk, remaining_chunk = self._split_chunk(current)
                    overlapped_chunks.append(first_chunk)

        return overlapped_chunks

    def _create_chunk_starting_with_overlap(self, chunk, overlap):
        if self.split_units == "word":
            current_chunk = overlap + " " + chunk
        elif self.split_units == "token":
            # For token-based splitting, combine at token level
            # at this point we know that the tokenizer is already initialized
            overlap_tokens = self.tiktoken_tokenizer.encode(overlap)  # type: ignore
            chunk_tokens = self.tiktoken_tokenizer.encode(chunk)  # type: ignore
            current_chunk = self.tiktoken_tokenizer.decode(overlap_tokens + chunk_tokens)  # type: ignore
        else:  # char
            current_chunk = overlap + chunk
        return current_chunk

    def _get_overlap(self, overlapped_chunks: List[str]) -> Tuple[str, str]:
        """Get the previous overlapped chunk instead of the original chunk."""
        prev_chunk = overlapped_chunks[-1]
        overlap_start = max(0, self._chunk_length(prev_chunk) - self.split_overlap)

        if self.split_units == "word":
            word_chunks = prev_chunk.split()
            overlap = " ".join(word_chunks[overlap_start:])
        elif self.split_units == "token":
            # For token-based splitting, handle overlap at token level
            # at this point we know that the tokenizer is already initialized
            tokens = self.tiktoken_tokenizer.encode(prev_chunk)  # type: ignore
            overlap_tokens = tokens[overlap_start:]
            overlap = self.tiktoken_tokenizer.decode(overlap_tokens)  # type: ignore
        else:  # char
            overlap = prev_chunk[overlap_start:]

        return overlap, prev_chunk

    def _chunk_length(self, text: str) -> int:
        """
        Get the length of the chunk in the specified units (words, characters, or tokens).

        :param text: The text to measure.
        :returns: The length of the text in the specified units.
        """
        if self.split_units == "word":
            words = [word for word in text.split(" ") if word]
            return len(words)
        elif self.split_units == "char":
            return len(text)
        else:  # token
            # at this point we know that the tokenizer is already initialized
            return len(self.tiktoken_tokenizer.encode(text))  # type: ignore

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
                # re. ignore: correct SentenceSplitter initialization is checked at the initialization of the component
                sentence_with_spans = self.nltk_tokenizer.split_sentences(text)  # type: ignore
                splits = [sentence["sentence"] for sentence in sentence_with_spans]
            else:
                # add escape "\" to the separator and wrapped it in a group so that it's included in the splits as well
                escaped_separator = re.escape(curr_separator)
                escaped_separator = f"({escaped_separator})"

                # split the text and merge every two consecutive splits, i.e.: the text and the separator after it
                splits = re.split(escaped_separator, text)
                splits = [
                    "".join([splits[i], splits[i + 1]]) if i < len(splits) - 1 else splits[i]
                    for i in range(0, len(splits), 2)
                ]

                # remove last split if it's empty
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
                    # process current_chunk
                    if current_chunk:  # keep the good splits
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # recursively handle splits that are too large
                    if self._chunk_length(split_text) > self.split_length:
                        if curr_separator == self.separators[-1]:
                            # tried last separator, can't split further, do a fixed-split based on word/character/token
                            fall_back_chunks = self._fall_back_to_fixed_chunking(split_text, self.split_units)
                            chunks.extend(fall_back_chunks)
                        else:
                            chunks.extend(self._chunk_text(split_text))

                    else:
                        current_chunk.append(split_text)
                        current_length += self._chunk_length(split_text)
                else:
                    current_chunk.append(split_text)
                    current_length += self._chunk_length(split_text)

            if current_chunk:
                chunks.append("".join(current_chunk))

            if self.split_overlap > 0:
                chunks = self._apply_overlap(chunks)

            if chunks:
                return chunks

        # if no separator worked, fall back to word- or character-level chunking
        return self._fall_back_to_fixed_chunking(text, self.split_units)

    def _fall_back_to_fixed_chunking(self, text: str, split_units: Literal["word", "char", "token"]) -> List[str]:
        """
        Fall back to a fixed chunking approach if no separator works for the text.

        Splits the text into smaller chunks based on the split_length and split_units attributes, either by words,
        characters, or tokens.

        :param text: The text to be split into chunks.
        :param split_units: The unit of the split_length parameter. It can be either "word", "char", or "token".
        :returns:
            A list of text chunks.
        """
        chunks = []

        if split_units == "word":
            words = re.findall(r"\S+|\s+", text)
            current_chunk = []
            current_length = 0

            for word in words:
                if word != " ":
                    current_chunk.append(word)
                    current_length += 1
                    if current_length == self.split_length and current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk.append(word)

            if current_chunk:
                chunks.append("".join(current_chunk))
        elif split_units == "char":
            for i in range(0, self._chunk_length(text), self.split_length):
                chunks.append(text[i : i + self.split_length])
        else:  # token
            # at this point we know that the tokenizer is already initialized
            tokens = self.tiktoken_tokenizer.encode(text)  # type: ignore
            for i in range(0, len(tokens), self.split_length):
                chunk_tokens = tokens[i : i + self.split_length]
                chunks.append(self.tiktoken_tokenizer.decode(chunk_tokens))  # type: ignore
        return chunks

    def _add_overlap_info(self, curr_pos: int, new_doc: Document, new_docs: List[Document]) -> None:
        prev_doc = new_docs[-1]
        overlap_length = self._chunk_length(prev_doc.content) - (curr_pos - prev_doc.meta["split_idx_start"])  # type: ignore
        if overlap_length > 0:
            prev_doc.meta["_split_overlap"].append({"doc_id": new_doc.id, "range": (0, overlap_length)})
            new_doc.meta["_split_overlap"].append(
                {
                    "doc_id": prev_doc.id,
                    "range": (
                        self._chunk_length(prev_doc.content) - overlap_length,  # type: ignore
                        self._chunk_length(prev_doc.content),  # type: ignore
                    ),
                }
            )

    def _run_one(self, doc: Document) -> List[Document]:
        chunks = self._chunk_text(doc.content)  # type: ignore # the caller already check for a non-empty doc.content
        chunks = chunks[:-1] if len(chunks[-1]) == 0 else chunks  # remove last empty chunk if it exists
        current_position = 0
        current_page = 1

        new_docs: List[Document] = []

        for split_nr, chunk in enumerate(chunks):
            new_doc = Document(content=chunk, meta=deepcopy(doc.meta))
            new_doc.meta["split_id"] = split_nr
            new_doc.meta["split_idx_start"] = current_position
            new_doc.meta["_split_overlap"] = [] if self.split_overlap > 0 else None

            # add overlap information to the previous and current doc
            if split_nr > 0 and self.split_overlap > 0:
                self._add_overlap_info(current_position, new_doc, new_docs)

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

        :raises RuntimeError: If the component wasn't warmed up but requires it for sentence splitting or tokenization.
        """
        if not self._is_warmed_up and ("sentence" in self.separators or self.split_units == "token"):
            raise RuntimeError(
                "The component RecursiveDocumentSplitter wasn't warmed up but requires it "
                "for sentence splitting or tokenization. Call 'warm_up()' before calling 'run()'."
            )
        docs = []
        for doc in documents:
            if not doc.content or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            docs.extend(self._run_one(doc))

        return {"documents": docs}
