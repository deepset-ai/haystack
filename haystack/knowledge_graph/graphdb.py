from pathlib import Path
from typing import Optional

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from requests.auth import HTTPBasicAuth

from haystack.knowledge_graph.base import BaseKnowledgeGraph


class GraphDBKnowledgeGraph(BaseKnowledgeGraph):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7200,
        username: str = "",
        password: str = "",
        index: Optional[str] = None,
    ):
        self.url = f"http://{host}:{port}"
        self.index = index
        self.username = username
        self.password = password

    def import_from_ttl_file(self, index: str, path: Path):
        url = f"{self.url}/repositories/{index}/statements"
        headers = {"Content-type": "application/x-turtle"}
        response = requests.post(
            url,
            headers=headers,
            data=open(path, "r", encoding="utf-8").read(),
            auth=HTTPBasicAuth(self.username, self.password),
        )
        if response.status_code > 299:
            raise Exception(response.text)

    def get_all_triples(self, index: Optional[str] = None) -> List[Triple]:
        query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(query=query, index=index)
        return results

    def query(self, query: str, index: Optional[str] = None):
        if self.index is None and index is None:
            raise Exception("Index name is required")
        index = index or self.index
        sparql = SPARQLWrapper(f"{self.url}/repositories/{index}")
        sparql.setCredentials(self.username, self.password)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = list(sparql.query().convert())
        return results
