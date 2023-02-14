from typing import Dict, Optional

import logging
from collections import defaultdict
from pathlib import Path

from rdflib import Graph

from haystack.document_stores import BaseKnowledgeGraph

logger = logging.getLogger(__name__)


class InMemoryKnowledgeGraph(BaseKnowledgeGraph):
    """
    In memory Knowledge graph store, based on rdflib.
    """

    def __init__(self, index: str = "document"):
        """
        Init the in memory knowledge graph

        :param index: name of the index
        """
        super().__init__()

        self.indexes: Dict[str, Graph] = defaultdict(dict)  # type: ignore [arg-type]
        self.index: str = index

    def create_index(self, index: Optional[str] = None):
        """
        Create a new index stored in memory

        :param index: name of the index
        """
        index = index or self.index
        if index not in self.indexes:
            self.indexes[index] = Graph()
        else:
            logger.warning("Index '%s' is already present.", index)

    def delete_index(self, index: Optional[str] = None):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        """
        index = index or self.index

        if index in self.indexes:
            del self.indexes[index]
            logger.info("Index '%s' deleted.", index)

    def import_from_ttl_file(self, path: Path, index: Optional[str] = None):
        """
        Load in memory an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file

        :param path: path to a .ttl containing a knowledge graph
        :param index: name of the index
        """
        index = index or self.index
        self.indexes[index].parse(path)

    def get_all_triples(self, index: Optional[str] = None):
        """
        Query the given in memory index for all its stored triples. Duplicates are not filtered.

        :param index: name of the index
        :return: all triples stored in the index
        """
        sparql_query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_subjects(self, index: Optional[str] = None):
        """
        Query the given in memory index for all its stored subjects. Duplicates are not filtered.

        :param index: name of the index
        :return: all subjects stored in the index
        """
        sparql_query = "SELECT ?s WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def get_all_predicates(self, index: Optional[str] = None):
        """
        Query the given in memory index for all its stored predicates. Duplicates are not filtered.

        :param index: name of the index
        :return: all predicates stored in the index
        """
        sparql_query = "SELECT ?p WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def _create_document_field_map(self) -> Dict:
        """
        There is no field mapping required
        """
        return {}

    def get_all_objects(self, index: Optional[str] = None):
        """
        Query the given in memory index for all its stored objects. Duplicates are not filtered.

        :param index: name of the index
        :return: all objects stored in the index
        """
        sparql_query = "SELECT ?o WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query, index=index)
        return results

    def query(self, sparql_query: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Execute a SPARQL query on the given in memory index

        :param sparql_query: SPARQL query that shall be executed
        :param index: name of the index
        :return: query result
        """
        index = index or self.index
        raw_results = self.indexes[index].query(sparql_query)

        if raw_results.askAnswer is not None:
            return raw_results.askAnswer
        else:
            formatted_results = []
            for b in raw_results.bindings:
                formatted_result = {}
                items = list(b.items())
                for item in items:
                    type_ = item[0].toPython()[1:]
                    uri = item[1].toPython()
                    formatted_result[type_] = {"type": "uri", "value": uri}
                formatted_results.append(formatted_result)
            return formatted_results
