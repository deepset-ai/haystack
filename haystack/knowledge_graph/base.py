from typing import Optional


class BaseKnowledgeGraph:
    def run(self, sparql_query: str, index: Optional[str] = None, **kwargs):
        result = self.query(query=sparql_query, index=index)
        output = {"sparql_result": result, **kwargs}
        return output, "output_1"
