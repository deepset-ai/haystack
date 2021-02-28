from typing import Set

from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph


class QueryExecutor:

    def __init__(self, knowledge_graph: GraphDBKnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def execute(self, query: Query):
        print(query.get_sparql_query())
        # remove debugging hack
        #if query.question_type is QuestionType.BooleanQuestion:
        #    return True
        # return result string of executing sparql_query or Boolean if query is BooleanQuery
        return self.knowledge_graph.query(query=query.get_sparql_query(), index="hp-test")
        # TODO remove debug hack
        #return self.knowledge_graph.query(query="ASK WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")

        #return self.knowledge_graph.query(query="SELECT ?uri WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")

    def has_result(self, triples: Set[Triple]):
        return self.execute(Query(question_type=QuestionType.BooleanQuestion, triples=triples))
