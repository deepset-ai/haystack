import logging
import subprocess
import time
from pathlib import Path

from haystack.graph_retriever.text_to_sparql import Text2SparqlRetriever
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph
from haystack.preprocessor.utils import fetch_archive_from_http

logger = logging.getLogger(__name__)


def tutorial10_knowledge_graph():
    # Let's first fetch some triples that we want to store in our knowledge graph
    # Here: exemplary triples from the wizarding world
    graph_dir = "../data/tutorial10_knowledge_graph/"
    s3_url = "https://fandom-qa.s3-eu-west-1.amazonaws.com/triples_and_config.zip"
    fetch_archive_from_http(url=s3_url, output_dir=graph_dir)

    # Fetch a pre-trained BART model that translates text queries to SPARQL queries
    model_dir = "../saved_models/tutorial10_knowledge_graph/"
    s3_url = "https://fandom-qa.s3-eu-west-1.amazonaws.com/saved_models/hp_v3.4.zip"
    fetch_archive_from_http(url=s3_url, output_dir=model_dir)

    LAUNCH_GRAPHDB = True

    # Start a GraphDB server
    if LAUNCH_GRAPHDB:
        logging.info("Starting GraphDB ...")
        status = subprocess.run(
            ['docker run -d -p 7200:7200 --name graphdb-instance-tutorial docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11'], shell=True
        )
        if status.returncode:
            status = subprocess.run(
                [
                    'docker start graphdb-instance-tutorial'],
                shell=True
            )
            if status.returncode:
                raise Exception("Failed to launch GraphDB. If you want to connect to an already running GraphDB instance"
                            "then set LAUNCH_GRAPHDB in the script to False.")
        time.sleep(5)

    # Initialize a knowledge graph connected to GraphDB and use "tutorial_10_index" as the name of the index
    kg = GraphDBKnowledgeGraph(index="tutorial_10_index")

    # Delete the index as it might have been already created in previous runs
    kg.delete_index()

    # Create the index based on a configuration file
    kg.create_index(config_path=Path(graph_dir+"repo-config.ttl"))

    # Import triples of subject, predicate, and object statements from a ttl file
    kg.import_from_ttl_file(index="tutorial_10_index", path=Path(graph_dir+"triples.ttl"))
    logging.info(f"The last triple stored in the knowledge graph is: {kg.get_all_triples()[-1]}")
    logging.info(f"There are {len(kg.get_all_triples())} triples stored in the knowledge graph.")

    # Define prefixes for names of resources so that we can use shorter resource names in queries
    prefixes = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX hp: <https://deepset.ai/harry_potter/>
    """
    kg.prefixes = prefixes

    # Load a pre-trained model that translates text queries to SPARQL queries
    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path=model_dir+"hp_v3.4")

    # We can now ask questions that will be answered by our knowledge graph!
    # One limitation though: our pre-trained model can only generate questions about resources it has seen during training.
    # Otherwise, it cannot translate the name of the resource to the identifier used in the knowledge graph.
    # E.g. "Harry" -> "hp:Harry_potter"

    query = "In which house is Harry Potter?"
    logging.info(f"Translating the text query \"{query}\" to a SPARQL query and executing it on the knowledge graph...")
    result = kgqa_retriever.retrieve(query=query)
    logging.info(result)
    # Correct SPARQL query: select ?a { hp:Harry_potter hp:house ?a . }
    # Correct answer: Gryffindor

    logging.info("Executing a SPARQL query with prefixed names of resources...")
    result = kgqa_retriever._query_kg(sparql_query="select distinct ?sbj where { ?sbj hp:job hp:Keeper_of_keys_and_grounds . }")
    logging.info(result)
    # Paraphrased question: Who is the keeper of keys and grounds?
    # Correct answer: Rubeus Hagrid

    logging.info("Executing a SPARQL query with full names of resources...")
    result = kgqa_retriever._query_kg(sparql_query="select distinct ?obj where { <https://deepset.ai/harry_potter/Hermione_granger> <https://deepset.ai/harry_potter/patronus> ?obj . }")
    logging.info(result)
    # Paraphrased question: What is the patronus of Hermione?
    # Correct answer: Otter


if __name__ == "__main__":
    tutorial10_knowledge_graph()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/