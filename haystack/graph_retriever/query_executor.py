import logging
from typing import Set, List, Union

from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


class QueryExecutor:

    def __init__(self, knowledge_graph: GraphDBKnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def execute(self, query: Query) -> Union[int, bool, List[str], None]:
        """
        return list of result string of executed sparql_query or Boolean if query has QuestionType BooleanQuestion
        """
        response = self.knowledge_graph.query(query=query.get_sparql_query(), index="hp-test")
        # logger.info(query)
        result = None
        if query.question_type == QuestionType.CountQuestion and response is not None:
            #text_result = [result_item["count_result"]["value"] for result_item in result]
            result = int(response[0]["count_result"]["value"])
        elif query.question_type == QuestionType.ListQuestion and response is not None:
            result = [result_item["uri"]["value"] if "uri" in result_item else "" for result_item in response]
        elif query.question_type == QuestionType.BooleanQuestion and response is not None:
            result = response
        return result

    def has_result(self, triples: Set[Triple]):
        return self.execute(Query(question_type=QuestionType.BooleanQuestion, triples=triples))
