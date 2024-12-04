import re
from typing import List

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class RecursiveChunker:
    def __init__(  # pylint: disable=too-many-positional-arguments
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
        self._check_params()
        if "sentence" in separators:
            self.nltk_tokenizer = self._get_custom_sentence_tokenizer()

    def _check_params(self):
        if self.chunk_overlap < 0:
            raise ValueError("Overlap must be greater than zero.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap cannot be greater than or equal to the chunk size.")

    @staticmethod
    def _get_custom_sentence_tokenizer():
        try:
            from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
        except (LookupError, ModuleNotFoundError):
            raise Exception("You need to install NLTK to use this function. You can install it via `pip install nltk`")
        return SentenceSplitter(language="en")

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Applies an overlap between consecutive chunks if the  chunk_overlap attribute is greater than zero.

        :param chunks: List of text chunks.
        :returns:
            The list of chunks with overlap applied.
        """
        overlapped_chunks = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
                continue
            overlap_start = max(0, len(chunks[idx - 1]) - self.chunk_overlap)
            current_chunk = chunks[idx - 1][overlap_start:] + chunk
            overlapped_chunks.append(current_chunk)
        return overlapped_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Recursive chunking algorithm that divides text into smaller chunks based on a list of separator characters.

        It starts with a list of separator characters (e.g., ["\n\n", "\n", " ", ""]) and attempts to divide the text
        using the first separator. If the resulting chunks are still larger than the specified chunk size, it moves to
        the next separator in the list.
        This process continues recursively, using progressively less specific separators until the chunks meet the
        desired size criteria.

        :param text:
        :returns:
            A list of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # try each separator
        for separator in self.separators:
            if separator in "sentence":
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
                if current_length + len(split_text) > self.chunk_size:
                    if current_chunk:  # keep the good splits
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

            if self.chunk_overlap > 0:
                chunks = self._apply_overlap(chunks)

            return chunks

        # If no separator worked, fall back to character-level chunking
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def _run_one(self, doc: Document) -> List[Document]:
        new_docs = []
        # NOTE: the check for a non-empty content is already done in the run method
        chunks = self._chunk_text(doc.content)  # type: ignore
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
            if not doc.content or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            new_docs.extend(self._run_one(doc))
        return new_docs
