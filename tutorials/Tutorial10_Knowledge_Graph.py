import logging
import subprocess
import time
from pathlib import Path

from farm.utils import initialize_device_settings

from haystack.graph_retriever.kgqa import Text2SparqlRetriever
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


def tutorial10_knowledge_graph():
    LAUNCH_GRAPHDB = False

    device, n_gpu = initialize_device_settings(use_cuda=True)

    # Start a GraphDB server
    if LAUNCH_GRAPHDB:
        logging.info("Starting GraphDB ...")
        status = subprocess.run(
            ['docker run -d -p 7200:7200 --name graphdb-instance-lcquad docker-registry.ontotext.com/graphdb-free:9.4.1-adoptopenjdk11'], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch GraphDB. If you want to connect to an existing GraphDB instance"
                            "then set LAUNCH_GRAPHDB in the script to False.")
        time.sleep(30)


    kg = GraphDBKnowledgeGraph(index="lcquad_full_wikidata")
    if LAUNCH_GRAPHDB:
        kg.create_index(config_path=Path("../data/repo-config.ttl"))
        kg.import_from_ttl_file(index="lcquad_full_wikidata", path=Path("../data/tutorial10_knowledge_graph.ttl"))
    #print(kg.get_all_triples()[:10])
    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../saved_models/lcquad_full_wikidata", top_k=1)

    result = kgqa_retriever.retrieve(question_text="What periodical literature does Delta Air Lines use as a moutpiece?")
    # SPARQL query: select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 }
    # Answer: wd:Q3486420

if __name__ == "__main__":
    tutorial10_knowledge_graph()
