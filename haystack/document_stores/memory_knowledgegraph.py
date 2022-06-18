from typing import Dict, Optional

from pathlib import Path

from haystack.document_stores import BaseKnowledgeGraph

from rdflib import Graph


class InMemoryKnowledgeGraph(BaseKnowledgeGraph):
    """
    In memory Knowledge graph store, based on rdflib.
    """

    def __init__(
        self,
        index: Optional[str] = None,
    ):
        """
        Init the in memory knowledge graph

        :param index: name of the index
        """
        super().__init__()

        self.index = index

    def create_index(self):
        """
        Create a new index stored in memory
        """

        if self.index:
            self.graph = Graph(identifier=self.index)
        else:
            self.graph = Graph()
            self.index = str(self.graph.identifier)

    def import_from_ttl_file(self, path: Path):
        """
        Load in memory an existing knowledge graph represented in the form of triples of subject, predicate, and object from a .ttl file

        :param path: path to a .ttl containing a knowledge graph
        """
        self.graph.parse(path)

    def get_all_triples(self):
        """
        Query the in memory index for all its stored triples. Duplicates are not filtered.

        :return: all triples stored in the index
        """
        sparql_query = "SELECT * WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query)
        return results

    def get_all_subjects(self):
        """
        Query the in memory index for all its stored subjects. Duplicates are not filtered.

        :return: all subjects stored in the index
        """
        sparql_query = "SELECT ?s WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query,)
        return results

    def get_all_predicates(self):
        """
        Query the in memory index for all its stored predicates. Duplicates are not filtered.

        :return: all predicates stored in the index
        """
        sparql_query = "SELECT ?p WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query)
        return results

    def _create_document_field_map(self) -> Dict:
        """
        There is no field mapping required
        """
        return {}

    def get_all_objects(self):
        """
        Query the in memory index for all its stored objects. Duplicates are not filtered.

        :return: all objects stored in the index
        """
        sparql_query = "SELECT ?o WHERE { ?s ?p ?o. }"
        results = self.query(sparql_query=sparql_query)
        return results

    def query(self, sparql_query: str):
        """
        Execute a SPARQL query on the given in memory index

        :param sparql_query: SPARQL query that shall be executed
        :return: query result
        """
        results = self.graph.query(sparql_query)

        if results.askAnswer is not None:
            return results.askAnswer
        else:
            formatted_results=[]
            for b in results.bindings:
                formatted_result={}
                items = list(b.items())
                for item in items:
                    type_ = item[0].toPython()[1:]
                    uri = item[1].toPython()
                    formatted_result[type_] = {'type':'uri', 'value':uri}
                formatted_results.append(formatted_result)
            return formatted_results