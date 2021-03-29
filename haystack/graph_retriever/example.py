import logging

from haystack.graph_retriever.kgqa import Text2SparqlRetriever
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


def run_experiments():
    kg = GraphDBKnowledgeGraph(index="lcquad_full_wikidata")
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)

    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../saved_models/lcquad_full_wikidata", top_k=1)

    result = kgqa_retriever.retrieve(question_text="What is your question?", top_k_graph=1)

if __name__ == "__main__":
    run_experiments()
