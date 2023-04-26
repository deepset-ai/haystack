from typing import Literal, Iterable

import re
import logging

import rank_bm25
from tqdm import tqdm

from haystack.preview.dataclasses import Document


logger = logging.getLogger(__name__)


class BM25Ranking:
    """
    Implementa BM25 ranking for MemoryDocumentStore. It's a thin wrapper on top of
    [`rank_bm25`](https://github.com/dorianbrown/rank_bm25), see its documentation for more details.
    """

    def __init__(
        self,
        progress_bar: bool = True,
        tokenization_regex: str = r"(?u)\b\w\w+\b",
        algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        **parameters,
    ):
        self.progress_bar = progress_bar
        self.tokenization_regex = tokenization_regex
        self.algorithm = algorithm
        self.parameters = parameters
        self.ranking: rank_bm25.BM25 = {}

    @property
    def tokenization_regex(self):
        return self._tokenizer

    @tokenization_regex.setter
    def tokenization_regex(self, regex_string: str):
        self._tokenizer = re.compile(regex_string).findall

    @property
    def algorithm(self):
        return self._bm25_class

    @algorithm.setter
    def algorithm(self, algorithm: str):
        self._bm25_class = getattr(rank_bm25, algorithm)

    def update_bm25(self, documents: Iterable[Document]) -> None:
        """
        Updates the BM25 sparse representation.

        :param documents: an iterable returning all the documents in the docstore
        """
        tokenized_corpus = []

        for doc in tqdm(documents, unit=" docs", desc="Updating BM25 ranking...", disable=not self.progress_bar):
            if doc.content_type != "text":
                logger.warning(
                    "Type of document %s is not 'text'. It won't be present in the BM25 ranking. "
                    "To silence this warning, consider using different document stores for different "
                    "document types, or switch to retrieval by embedding.",
                    doc.id,
                )
            else:
                tokenized_corpus.append(self.tokenization_regex(doc.content.lower()))
        self.ranking = self.algorithm(tokenized_corpus, **self.parameters)
