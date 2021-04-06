from haystack import BaseComponent
from haystack.knowledge_graph.base import BaseKnowledgeGraph


class BaseGraphRetriever(BaseComponent):
    knowledge_graph: BaseKnowledgeGraph
    outgoing_edges = 1

    def retrieve(self, question_text: str, top_k: int):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self, query, top_k, **kwargs):
        raise NotImplementedError
