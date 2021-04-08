from pathlib import Path
from typing import Optional

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from requests.auth import HTTPBasicAuth

from haystack.knowledge_graph.base import BaseKnowledgeGraph


class GraphDBKnowledgeGraph(BaseKnowledgeGraph):
    """
        Knowledge graph store that runs on a GraphDB instance
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7200,
        username: str = "",
        password: str = "",
        index: Optional[str] = None,
        prefixes: str = ""
    ):
        self.url = f"http://{host}:{port}"
        self.index = index
        self.username = username
        self.password = password
        self.prefixes = prefixes

    def create_index(self, config_path: Path):
        url = f"{self.url}/rest/repositories"
        files = {'config': open(config_path, "r", encoding="utf-8")}
        response = requests.post(
            url,
            files=files,
        )
        if response.status_code > 299:
            raise Exception(response.text)

    def delete_index(self):
        url = f"{self.url}/rest/repositories/{self.index}"
        response = requests.delete(url)
        if response.status_code > 299:
            raise Exception(response.text)

    def import_from_ttl_file(self, index: str, path: Path):
        url = f"{self.url}/repositories/{index}/statements"
        headers = {"Content-type": "application/x-turtle"}
        response = requests.post(
            url,
            headers=headers,
            data=open(path, "r", encoding="utf-8").read().encode('utf-8'),
            auth=HTTPBasicAuth(self.username, self.password),
        )
        if response.status_code > 299:
            raise Exception(response.text)

    def get_all_triples(self, index: Optional[str] = None):
        sparql_query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_subjects(self, index: Optional[str] = None):
        sparql_query = "SELECT ?s WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_predicates(self, index: Optional[str] = None):
        sparql_query = "SELECT ?p WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_objects(self, index: Optional[str] = None):
        sparql_query = "SELECT ?o WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def query(self, sparql_query: str, index: Optional[str] = None):
        if self.index is None and index is None:
            raise Exception("Index name is required")
        index = index or self.index
        sparql = SPARQLWrapper(f"{self.url}/repositories/{index}")
        sparql.setCredentials(self.username, self.password)
        sparql.setQuery(self.prefixes + sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # if query is a boolean query, return boolean instead of text result
        return results["results"]["bindings"] if "results" in results else results["boolean"]
