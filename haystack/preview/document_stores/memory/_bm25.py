from typing import Literal, Generator

import re
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import rank_bm25
except ImportError as e:
    logger.debug("rank_bm25 not found: MemoryDocumentStore won't be able to retrieve by BM25.")


from haystack.preview.dataclasses import Document
from haystack.preview.document_stores._utils import StoreError


class BM25RepresentationMissing(StoreError):
    pass


class BM25Representation:
    def __init__(
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: dict = {},
    ):
        self.bm25_tokenization_regex = bm25_tokenization_regex
        self.bm25_algorithm = bm25_algorithm
        self.bm25_parameters = bm25_parameters
        self.bm25_ranking: rank_bm25.BM25 = {}

    @property
    def bm25_tokenization_regex(self):
        return self._tokenizer

    @bm25_tokenization_regex.setter
    def bm25_tokenization_regex(self, regex_string: str):
        self._tokenizer = re.compile(regex_string).findall

    @property
    def bm25_algorithm(self):
        return self._bm25_class

    @bm25_algorithm.setter
    def bm25_algorithm(self, algorithm: str):
        self._bm25_class = getattr(rank_bm25, algorithm)

    def update_bm25(self, documents: Generator[Document, None, None]) -> None:
        """
        Updates the BM25 sparse representation in the the document store.

        :param documents: a generator returning all the documents in the docstore
        """
        tokenized_corpus = []

        # TODO Enable/disable progress bar
        for doc in tqdm(documents, unit=" docs", desc="Updating BM25 representation..."):
            if doc.content_type != "text":
                logger.warning("Document %s is non-textual. It won't be present in the BM25 pool.", doc.id)
            else:
                tokenized_corpus.append(self.bm25_tokenization_regex(doc.content.lower()))

        self.bm25_ranking = self.bm25_algorithm(tokenized_corpus, **self.bm25_parameters)
