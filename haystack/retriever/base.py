from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query, candidate_doc_ids=None, top_k=1):
        pass
