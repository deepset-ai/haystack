import logging
from typing import Optional, Set

from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple

logger = logging.getLogger(__name__)


class Query:

    def __init__(self, question_type: QuestionType, triples: Set[Triple]):
        self.triples: Set[Triple] = triples
        self.question_type: QuestionType = question_type
        self._sparql_query: Optional[str] = None
        self._verbalized_sparql_query: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.get_sparql_query()}"

    def has_variable_in_every_triple(self) -> bool:
        for triple in self.triples:
            if not triple.has_variable():
                return False
        return True

    def has_uri_variable(self) -> bool:
        for triple in self.triples:
            if triple.has_uri_variable():
                return True
        return False

    def get_sparql_query(self) -> str:
        if not self._sparql_query:
            self._sparql_query = self.build_sparql_query_string()
        return self._sparql_query

    def get_verbalized_sparql_query(self) -> str:
        """
        Replace identifiers of entities and relations in the sparql query with their natural language label
        """
        sparql_query = self.get_sparql_query()
        verbalized_query = sparql_query.replace("<https://deepset.ai/harry_potter/", "").replace("_", " ").replace(">", "")
        return verbalized_query

    def build_sparql_query_string(self) -> str:
        """
        Combine triples in one where_clause and generate a SPARQL query based on a template for one of the three question types
        """
        where_clause = self.build_where_clause()
        query = None
        if self.question_type == QuestionType.CountQuestion:
            query = f"SELECT (COUNT(  ?uri ) AS ?count_result) WHERE {{ {where_clause} }}"
        elif self.question_type == QuestionType.BooleanQuestion:
            query = f"ASK WHERE {{ {where_clause} }}"
        elif self.question_type == QuestionType.ListQuestion:
            query = f"SELECT ?uri WHERE {{ {where_clause} }}"

        if not query:
            raise RuntimeError(f"QuestionType {self.question_type} unknown")

        return query

    def build_where_clause(self):
        """
        Combine triples in one where_clause
        """
        triples_text = [str(triple) for triple in self.triples]
        where_clause = ". ".join(triples_text)
        return where_clause
