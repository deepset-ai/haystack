from typing import Dict, Optional, Union, Tuple

from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "graphdb", ie)

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
        prefixes: str = "",
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
        super().__init__()

        self.url = f"http://{host}:{port}"
        self.index = index
        self.username = username
        self.password = password
        self.prefixes = prefixes

    def create_index(
        self,
        config_path: Path,
        headers: Optional[Dict[str, str]] = None,
        timeout: Union[float, Tuple[float, float]] = 10.0,
    ):
        """
        Create a new index (also called repository) stored in the GraphDB instance

        :param config_path: path to a .ttl file with configuration settings, details:
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :param timeout: How many seconds to wait for the server to send data before giving up,
            as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
            Defaults to 10 seconds.
        https://graphdb.ontotext.com/documentation/free/configuring-a-repository.html#configure-a-repository-programmatically
        """
        url = f"{self.url}/rest/repositories"
        files = {"config": open(config_path, "r", encoding="utf-8")}
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
        if response.status_code > 299:
            raise Exception(response.text)

    def delete_index(self, headers: Optional[Dict[str, str]] = None, timeout: Union[float, Tuple[float, float]] = 10.0):
        """
        Delete the index that GraphDBKnowledgeGraph is connected to. This method deletes all data stored in the index.
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :param timeout: How many seconds to wait for the server to send data before giving up,
            as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
            Defaults to 10 seconds.
        """
        url = f"{self.url}/rest/repositories/{self.index}"
        response = requests.delete(url, headers=headers, timeout=timeout)
        if response.status_code > 299:
            raise Exception(response.text)

    def import_from_ttl_file(
        self,
        index: str,
        path: Path,
        headers: Optional[Dict[str, str]] = None,
        timeout: Union[float, Tuple[float, float]] = 10.0,
    ):
        """
        Load an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file into an index of GraphDB

        :param index: name of the index (also called repository) in the GraphDB instance where the imported triples shall be stored
        :param path: path to a .ttl containing a knowledge graph
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :param timeout: How many seconds to wait for the server to send data before giving up,
            as a float, or a :ref:`(connect timeout, read timeout) <timeouts>` tuple.
            Defaults to 10 seconds.
        """
        url = f"{self.url}/repositories/{index}/statements"
        headers = (
            {"Content-type": "application/x-turtle"}
            if headers is None
            else {**{"Content-type": "application/x-turtle"}, **headers}
        )
        response = requests.post(
            url,
            headers=headers,
            data=open(path, "r", encoding="utf-8").read().encode("utf-8"),
            auth=HTTPBasicAuth(self.username, self.password),
            timeout=timeout,
        )
        if response.status_code > 299:
            raise Exception(response.text)

    def get_all_triples(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Query the given index in the GraphDB instance for all its stored triples. Duplicates are not filtered.

        :param index: name of the index (also called repository) in the GraphDB instance
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :return: all triples stored in the index
        """
        sparql_query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index, headers=headers)
        return results

    def get_all_subjects(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Query the given index in the GraphDB instance for all its stored subjects. Duplicates are not filtered.

        :param index: name of the index (also called repository) in the GraphDB instance
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :return: all subjects stored in the index
        """
        sparql_query = "SELECT ?s WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index, headers=headers)
        return results

    def get_all_predicates(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Query the given index in the GraphDB instance for all its stored predicates. Duplicates are not filtered.

        :param index: name of the index (also called repository) in the GraphDB instance
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :return: all predicates stored in the index
        """
        sparql_query = "SELECT ?p WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index, headers=headers)
        return results

    def _create_document_field_map(self) -> Dict:
        """
        There is no field mapping required
        """
        return {}

    def get_all_objects(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Query the given index in the GraphDB instance for all its stored objects. Duplicates are not filtered.

        :param index: name of the index (also called repository) in the GraphDB instance
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :return: all objects stored in the index
        """
        sparql_query = "SELECT ?o WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index, headers=headers)
        return results

    def query(self, sparql_query: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Execute a SPARQL query on the given index in the GraphDB instance

        :param sparql_query: SPARQL query that shall be executed
        :param index: name of the index (also called repository) in the GraphDB instance
        :param headers: Custom HTTP headers to pass to http client (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='})
        :return: query result
        """
        if self.index is None and index is None:
            raise Exception("Index name is required")
        index = index or self.index
        sparql = SPARQLWrapper(f"{self.url}/repositories/{index}")
        sparql.setCredentials(self.username, self.password)
        sparql.setQuery(self.prefixes + sparql_query)
        sparql.setReturnFormat(JSON)
        if headers is not None:
            sparql.customHttpHeaders = headers
        results = sparql.query().convert()
        # if query is a boolean query, return boolean instead of text result
        # FIXME: 'results' likely doesn't support membership test (`"something" in results`).
        # Pylint raises unsupported-membership-test and unsubscriptable-object.
        # Silenced for now, keep in mind for future debugging.
        return (
            results["results"]["bindings"]  # type: ignore  # pylint: disable=unsubscriptable-object
            if "results" in results  # type: ignore  # pylint: disable=unsupported-membership-test
            else results["boolean"]  # type: ignore  # pylint: disable=unsubscriptable-object
        )
