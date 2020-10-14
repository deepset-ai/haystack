import logging
from typing import List

from haystack import Document
from haystack.generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class RAGGenerator(BaseGenerator):

    def __init__(
            self,
            model: str = "facebook/rag-token-nq",
            use_gpu: int = 0,
            return_no_answers: bool = True,
            max_seq_len: int = 100
    ):
        pass

    def generate(self, question: str, documents: List[Document]):
        if not self.tokenizer:
            raise AttributeError("RAGGenerator: generate function need self.tokenizer")

        pass
