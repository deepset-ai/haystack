from typing import Set

from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph


class QueryExecutor:

    def __init__(self, knowledge_graph: GraphDBKnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def execute(self, query: Query):
        """
        return result string of executed sparql_query or Boolean if query has QuestionType BooleanQuestion
        """
        # print(query.get_sparql_query())
        return self.knowledge_graph.query(query=query.get_sparql_query(), index="hp-test")

    def has_result(self, triples: Set[Triple]):
        return self.execute(Query(question_type=QuestionType.BooleanQuestion, triples=triples))
