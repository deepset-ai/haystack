import logging
import subprocess
import time
from pathlib import Path

from farm.utils import initialize_device_settings

from haystack.graph_retriever.kgqa import Text2SparqlRetriever
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


def graph_retrieval():
    LAUNCH_GRAPHDB = False

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)

    device, n_gpu = initialize_device_settings(use_cuda=True)

    # Start a GraphDB server
    if LAUNCH_GRAPHDB:
        logging.info("Starting GraphDB ...")
        status = subprocess.run(
            ['docker run -d -p 7200:7200 --name graphdb-instance-lcquad ontotext/graphdb:9.7.0-se'], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch GraphDB. If you want to connect to an existing GraphDB instance"
                            "then set LAUNCH_GRAPHDB in the script to False.")
        time.sleep(30)


    kg = GraphDBKnowledgeGraph(index="lcquad_full_wikidata")
    kg.create_index(config_path="../../data/repo-config.ttl")
    kg.import_from_ttl_file(index="lcquad_full_wikidata", path=Path("../../data/lcquad_example1.ttl"))
    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../saved_models/lcquad_full_wikidata", top_k=1)

    result = kgqa_retriever.retrieve(question_text="What is your question?")

if __name__ == "__main__":
    graph_retrieval()
