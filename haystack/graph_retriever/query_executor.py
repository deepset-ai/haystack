import logging
from typing import Set

from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)

class QueryExecutor:

    def __init__(self, knowledge_graph: GraphDBKnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def execute(self, query: Query):
        """
        return result string of executed sparql_query or Boolean if query has QuestionType BooleanQuestion
        """
        result = self.knowledge_graph.query(query=query.get_sparql_query(), index="hp-test")
        text_result = ""
        if query.question_type == QuestionType.CountQuestion and result is not None:
            text_result = str(result[0]["count_result"]["value"])
        elif query.question_type == QuestionType.ListQuestion and result is not None:
            text_result = result[0]["uri"]["value"]
        elif query.question_type == QuestionType.BooleanQuestion and result is not None:
            text_result = result
        return text_result

    def has_result(self, triples: Set[Triple]):
        return self.execute(Query(question_type=QuestionType.BooleanQuestion, triples=triples))
