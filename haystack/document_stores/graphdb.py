from typing import Optional

import requests
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
from requests.auth import HTTPBasicAuth

from haystack.document_stores import BaseKnowledgeGraph


class GraphDBKnowledgeGraph(BaseKnowledgeGraph):
    """
    Knowledge graph store that runs on a GraphDB instance.
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
        """
        Init the knowledge graph by defining the settings to connect with a GraphDB instance
        
        :param host: address of server where the GraphDB instance is running
        :param port: port where the GraphDB instance is running
        :param username: username to login to the GraphDB instance (if any)
        :param password: password to login to the GraphDB instance (if any)
        :param index: name of the index (also called repository) stored in the GraphDB instance
        :param prefixes: definitions of namespaces with a new line after each namespace, e.g., PREFIX hp: <https://deepset.ai/harry_potter/>
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
            host=host, port=port, username=username, password=password, index=index, prefixes=prefixes
        )

        self.url = f"http://{host}:{port}"
        self.index = index
        self.username = username
        self.password = password
        self.prefixes = prefixes

    def create_index(self, config_path: Path):
        """
        Create a new index (also called repository) stored in the GraphDB instance
        
        :param config_path: path to a .ttl file with configuration settings, details: 
        https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically
        """
        url = f"{self.url}/rest/repositories"
        files = {'config': open(config_path, "r", encoding="utf-8")}
        response = requests.post(
            url,
            files=files,
        )
        if response.status_code > 299:
            raise Exception(response.text)

    def delete_index(self):
        """
        Delete the index that GraphDBKnowledgeGraph is connected to. This method deletes all data stored in the index.
        """
        url = f"{self.url}/rest/repositories/{self.index}"
        response = requests.delete(url)
        if response.status_code > 299:
            raise Exception(response.text)

    def import_from_ttl_file(self, index: str, path: Path):
        """
        Load an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file into an index of GraphDB
        
        :param index: name of the index (also called repository) in the GraphDB instance where the imported triples shall be stored
        :param path: path to a .ttl containing a knowledge graph
        """
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
        """
        Query the given index in the GraphDB instance for all its stored triples. Duplicates are not filtered.
        
        :param index: name of the index (also called repository) in the GraphDB instance
        :return: all triples stored in the index
        """
        sparql_query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_subjects(self, index: Optional[str] = None):
        """
        Query the given index in the GraphDB instance for all its stored subjects. Duplicates are not filtered.
        
        :param index: name of the index (also called repository) in the GraphDB instance
        :return: all subjects stored in the index
        """ 
        sparql_query = "SELECT ?s WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_predicates(self, index: Optional[str] = None):
        """
        Query the given index in the GraphDB instance for all its stored predicates. Duplicates are not filtered.
        
        :param index: name of the index (also called repository) in the GraphDB instance
        :return: all predicates stored in the index
        """
        sparql_query = "SELECT ?p WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_objects(self, index: Optional[str] = None):
        """
        Query the given index in the GraphDB instance for all its stored objects. Duplicates are not filtered.
        
        :param index: name of the index (also called repository) in the GraphDB instance
        :return: all objects stored in the index
        """
        sparql_query = "SELECT ?o WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def query(self, sparql_query: str, index: Optional[str] = None):
        """
        Execute a SPARQL query on the given index in the GraphDB instance
        
        :param sparql_query: SPARQL query that shall be executed
        :param index: name of the index (also called repository) in the GraphDB instance
        :return: query result
        """
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
