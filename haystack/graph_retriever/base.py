from haystack.knowledge_graph.base import BaseKnowledgeGraph


class BaseGraphRetriever:
    knowledge_graph: BaseKnowledgeGraph
    outgoing_edges = 1

    def retrieve(self, question_text, top_k_graph):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self, query, top_k_graph, **kwargs):
        raise NotImplementedError
