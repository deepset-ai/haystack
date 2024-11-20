from typing import List

from haystack import component


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

    @staticmethod
    def _chunk_text(text):
        # some logic to split text into smaller chunks
        return text

    def run(self, documents):
        """
        Split text of documents into smaller chunks recursively.

        :param documents:
        :returns:
            Documents with text split into smaller chunks
        """
        for doc in documents:
            doc.text = self._chunk_text(doc.text)
        return documents
