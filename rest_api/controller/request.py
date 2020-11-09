import sys
from typing import Any, Collection, Dict, List, Optional, Union

from pydantic import BaseModel

from rest_api.config import DEFAULT_TOP_K_READER, DEFAULT_TOP_K_RETRIEVER

MAX_RECURSION_DEPTH = sys.getrecursionlimit() - 1


class Question(BaseModel):
    questions: List[str]
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None
    top_k_reader: int = DEFAULT_TOP_K_READER
    top_k_retriever: int = DEFAULT_TOP_K_RETRIEVER

    @classmethod
    def from_elastic_query_dsl(cls, query_request: Dict[str, Any], top_k_reader: int = DEFAULT_TOP_K_READER):

        # Refer Query DSL
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html
        # Currently do not support query matching with field parameter
        query_strings: List[str] = []
        filters: Dict[str, str] = {}
        top_k_retriever: int = DEFAULT_TOP_K_RETRIEVER if "size" not in query_request else query_request["size"]

        cls._iterate_dsl_request(query_request, query_strings, filters)

        if len(query_strings) != 1:
            raise SyntaxError('Only one valid `query` field required expected, '
                              'refer https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-match-query.html')

        return cls(questions=query_strings, filters=filters if len(filters) else None, top_k_retriever=top_k_retriever,
                   top_k_reader=top_k_reader)

    @classmethod
    def _iterate_dsl_request(cls, query_dsl: Any, query_strings: List[str], filters: Dict[str, str], depth: int = 0):
        if depth == MAX_RECURSION_DEPTH:
            raise RecursionError('Parsing incoming DSL reaching current value of the recursion limit')

        # For question: Only consider values of "query" key for "match" and "multi_match" request.
        # For filter: Only consider Dict[str, str] value of "term" or "terms" key
        if isinstance(query_dsl, List):
            for item in query_dsl:
                cls._iterate_dsl_request(item, query_strings, filters, depth + 1)
        elif isinstance(query_dsl, Dict):
            for key, value in query_dsl.items():
                # "query" value should be "str" type
                if key == 'query' and isinstance(value, str):
                    query_strings.append(value)
                elif key in ["filter", "filters"]:
                    cls._iterate_filters(value, filters, depth + 1)
                elif isinstance(value, Collection):
                    cls._iterate_dsl_request(value, query_strings, filters, depth + 1)

    @classmethod
    def _iterate_filters(cls, filter_dsl: Any, filters: Dict[str, str], depth: int = 0):
        if depth == MAX_RECURSION_DEPTH:
            raise RecursionError('Parsing incoming DSL reaching current value of the recursion limit')

        if isinstance(filter_dsl, List):
            for item in filter_dsl:
                cls._iterate_filters(item, filters, depth + 1)
        elif isinstance(filter_dsl, Dict):
            for key, value in filter_dsl.items():
                if key in ["term", "terms"]:
                    if isinstance(value, Dict):
                        for filter_key, filter_value in value.items():
                            # Currently only accepting Dict[str, str]
                            if isinstance(filter_value, str):
                                filters[filter_key] = filter_value
                elif isinstance(value, Collection):
                    cls._iterate_filters(value, filters, depth + 1)
