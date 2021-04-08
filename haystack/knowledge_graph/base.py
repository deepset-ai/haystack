from typing import Optional

from haystack import BaseComponent


class BaseKnowledgeGraph(BaseComponent):
    outgoing_edges = 1

    def run(self, sparql_query: str, index: Optional[str] = None, **kwargs):  # type: ignore
        result = self.query(sparql_query=sparql_query, index=index)
        output = {"sparql_result": result, **kwargs}
        return output, "output_1"

    def query(self, sparql_query: str, index: Optional[str] = None):
        raise NotImplementedError
