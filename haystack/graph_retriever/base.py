from haystack import BaseComponent
from haystack.knowledge_graph.base import BaseKnowledgeGraph

from abc import abstractmethod

class BaseGraphRetriever(BaseComponent):
    knowledge_graph: BaseKnowledgeGraph
    outgoing_edges = 1

    @abstractmethod
    def retrieve(self, question_text, top_k_graph):
        pass

    @abstractmethod
    def eval(self, filename, question_type, top_k_graph):
        pass
    @abstractmethod
    def run(self, query, top_k_graph, **kwargs):
        pass
