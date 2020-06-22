from abc import ABC, abstractmethod
from typing import List, Optional

from haystack.database.base import Document


class BaseReader(ABC):
    return_no_answers: Optional[bool]

    @abstractmethod
    def predict(self, question: str, documents: List[Document], top_k: Optional[int] = None):
        pass

    @abstractmethod
    def predict_batch(self, question_doc_list: List[dict], top_k_per_question: Optional[int] = None,
                      batch_size: Optional[int] = None):
        pass
