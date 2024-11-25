import re
from typing import List

from haystack import Document, component


@component
class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # Add overlap from previous chunk
                overlap_start = max(0, len(chunks[i - 1]) - self.chunk_overlap)
                current_chunk = chunks[i - 1][overlap_start:] + chunks[i]
                overlapped_chunks.append(current_chunk)
            else:
                overlapped_chunks.append(chunks[i])
        return overlapped_chunks

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        for separator in self.separators:
            # split using the current separator
            splits = text.split(separator) if not self.is_separator_regex else re.split(separator, text)

            # filter out empty splits
            splits = [s for s in splits if s.strip()]

            if len(splits) == 1:  # go to next separator, if current separator not found
                continue

            chunks = []
            current_chunk = []
            current_length = 0

            # check splits, if any is too long, recursively chunk it, otherwise add to current chunk
            for split in splits:
                split_text = split
                if self.keep_separator:
                    split_text = separator + split if split != splits[0] else split

                # if adding this split exceeds chunk_size, process current_chunk
                if current_length + len(split_text) > self.chunk_size:
                    if current_chunk:  # Save the good splits
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # recursively handle splits that are too large
                    if len(split_text) > self.chunk_size:
                        chunks.extend(self._chunk_text(split_text))
                    else:
                        chunks.append(split_text)
                else:
                    current_chunk.append(split_text)
                    current_length += len(split_text)

            if current_chunk:
                chunks.append("".join(current_chunk))

            chunks = self._apply_overlap(chunks)

            return chunks

        # If no separator worked, fall back to character-level chunking
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def _run_one(self, doc: Document) -> List[Document]:
        new_docs = []
        chunks = self._chunk_text(doc.content)
        for chunk in chunks:
            new_doc = Document(content=chunk, meta=doc.meta)
            new_doc.meta["original_id"] = doc.id
            new_docs.append(new_doc)
        return new_docs

    def run(self, documents: List[Document]) -> List[Document]:
        """
        Split text of documents into smaller chunks recursively.
        """
        new_docs = []
        for doc in documents:
            new_docs.extend(self._run_one(doc))
        return new_docs
