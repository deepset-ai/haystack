import logging
import subprocess
import time
from pathlib import Path

from haystack.graph_retriever.kgqa import Text2SparqlRetriever
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph
from haystack.preprocessor.utils import fetch_archive_from_http

logger = logging.getLogger(__name__)


def tutorial10_knowledge_graph():
    # Let's first fetch some triples that we want to store in our knowledge graph
    # Here: exemplary triples from the LC-QuAD 2.0 dataset
    doc_dir = "../data/tutorial10_knowledge_graph/"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/tutorial10_knowledge_graph.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Fetch a pre-trained model that translates natural language questions to SPARQL queries
    doc_dir = "../saved_models/tutorial10_knowledge_graph/"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/research/lcquad_wikidata.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    LAUNCH_GRAPHDB = True

    # Start a GraphDB server
    if LAUNCH_GRAPHDB:
        logging.info("Starting GraphDB ...")
        status = subprocess.run(
            ['docker run -d -p 7200:7200 --name graphdb-instance-lcquad docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11'], shell=True
        )
        if status.returncode:
            status = subprocess.run(
                [
                    'docker start graphdb-instance-lcquad'],
                shell=True
            )
            if status.returncode:
                raise Exception("Failed to launch GraphDB. If you want to connect to an existing GraphDB instance"
                            "then set LAUNCH_GRAPHDB in the script to False.")
        time.sleep(5)


    # Initialize a knowledge graph connected to GraphDB and use "lcquad_full_wikidata" as the name of the index
    kg = GraphDBKnowledgeGraph(index="lcquad_full_wikidata")

    # Delete the index as it might have been already created in previous runs
    kg.delete_index()

    # Create the index based on a configuration file
    kg.create_index(config_path=Path("../data/tutorial10_knowledge_graph/repo-config.ttl"))

    # Import triples of subject, predicate, and object statements from a ttl file
    kg.import_from_ttl_file(index="lcquad_full_wikidata", path=Path("../data/tutorial10_knowledge_graph/tutorial10_knowledge_graph.ttl"))
    logging.info(f"The last triple stored in the knowledge graph is: {kg.get_all_triples()[-1]}")
    logging.info(f"There are {len(kg.get_all_triples())} triples stored in the knowledge graph.")

    # Define prefixes for names of resources so that we can use shorter resource names in queries
    prefixes = """PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX cc: <http://creativecommons.org/ns#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX p: <http://www.wikidata.org/prop/>
    PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
    PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
    PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
    PREFIX pr: <http://www.wikidata.org/prop/reference/>
    PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
    PREFIX ps: <http://www.wikidata.org/prop/statement/>
    PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
    PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
    PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
    PREFIX wdref: <http://www.wikidata.org/reference/>
    PREFIX wds: <http://www.wikidata.org/entity/statement/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
    PREFIX wdv: <http://www.wikidata.org/value/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """
    kg.prefixes = prefixes

    # Load a pre-trained model that translates natural language questions to SPARQL queries
    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../saved_models/tutorial10_knowledge_graph/lcquad_full_wikidata")

    # We can now ask questions that will be answered by our knowledge graph!
    # One limitation though: our pre-trained model can only generate questions about resources it has seen during training.
    # Otherwise it cannot translate the name of the resource to the identifier used in the knowledge graph.
    # E.g. "Delta Air Lines" -> "wd:Q188920"
    question_text = "What periodical literature does Delta Air Lines use as a mouthpiece?"
    logging.info(f"Translating the natural language question \"{question_text}\" to a SPARQL query and executing it on the knowledge graph...")
    result = kgqa_retriever.retrieve(question_text=question_text)
    logging.info(result)
    # Correct SPARQL query: select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 }
    # Correct answer: wd:Q3486420

    logging.info("Executing a SPARQL query with prefixed names of resources...")
    result = kgqa_retriever._query_kg(query="select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . }")
    logging.info(result)
    # Correct answer:  http://www.wikidata.org/entity/Q3486420
    # https://query.wikidata.org/#select%20distinct%20%3Fobj%20where%20%7B%20wd%3AQ188920%20wdt%3AP2813%20%3Fobj%20.%7D

    logging.info("Executing a SPARQL query with full names of resources...")
    result = kgqa_retriever._query_kg(query="select distinct ?obj where { <http://www.wikidata.org/entity/Q188920> <http://www.wikidata.org/prop/direct/P2813> ?obj . }")
    logging.info(result)
    # Correct answer:  http://www.wikidata.org/entity/Q3486420


if __name__ == "__main__":
    tutorial10_knowledge_graph()
