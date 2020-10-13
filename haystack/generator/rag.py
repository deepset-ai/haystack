import logging
from pathlib import Path
from typing import List, Union

from haystack import Document
from haystack.generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class RAGGenerator(BaseGenerator):

    def __init__(
            self,
            model_name_or_path: Union[str, Path],
            max_seq_len: int = 100
    ):
        pass

    def predict(self, question: str, documents: List[Document]):
        pass
