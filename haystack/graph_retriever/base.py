from abc import abstractmethod

from haystack import BaseComponent
from haystack.knowledge_graph.base import BaseKnowledgeGraph


class BaseGraphRetriever(BaseComponent):
    knowledge_graph: BaseKnowledgeGraph
    outgoing_edges = 1

    @abstractmethod
    def retrieve(self, query: str, top_k: int):
        pass

    def eval(self):
        raise NotImplementedError

    def run(self, query: str, top_k: int):  # type: ignore
        answers = self.retrieve(query=query, top_k=top_k)
        results = {"answers": answers}
        return results, "output_1"
