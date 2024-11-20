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
        # 1. identify all occurrences of the first splitting character in the text

        # 2. split the text at the first occurrence of the splitting character

        # 3. assessing each split to check whether they meet the condition of being smaller than our specified chunk
        #    size

        # 4. splits that satisfy this condition can be labeled as good splits.
        # 4.1 combine good splits if each individual split is smaller than the chunk size

        # 5. splits that don't satisfy the condition of being smaller than the chunk size can be labeled as bad splits
        # 5.1 split the bad splits recursively until they meet the condition of being smaller than the chunk size

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
