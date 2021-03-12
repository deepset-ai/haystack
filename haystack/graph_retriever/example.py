import logging

import pandas as pd

from haystack.graph_retriever.kgqa import KGQARetriever, Text2SparqlRetriever
from haystack.graph_retriever.utils import eval_on_all_data
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph
from haystack import Pipeline

logger = logging.getLogger(__name__)


def run_examples(kgqa_retriever: KGQARetriever, top_k_graph: int):
    result = kgqa_retriever.retrieve(question_text="What is the hair color of Hermione?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Albus Dumbledore die?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who has Blond hair?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Harry die?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the patronus of Harry?", top_k=top_k_graph)


def run_experiments():
    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)

    # settings
    # kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="xxx")
    # kg.index = "hp-test"
    input_df = pd.read_csv("data/all_questions.csv")
    #input_df = pd.read_csv("../../data/harry/test.csv")

    # kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../../models/kgqa/hp_v3.2")
    # kgqa_retriever = KGQARetriever(knowledge_graph=kg,
    #                                query_ranker_path="saved_models/lcquad_text_pair_classification_with_entity_labels_v2",
    #                                alias_to_entity_and_prob_path="alias_to_entity_and_prob.json",
    #                                token_and_relation_to_tfidf_path="token_and_relation_to_tfidf.json")
    top_k_graph = 10

    p = Pipeline.load_from_yaml("config/pipelines.yaml")
    results = eval_on_all_data(p, top_k_graph=top_k_graph, input_df=input_df, query_executor="both")
    results.to_csv("data/results.csv", index=False)

    #
    # top_k_graph = 1
    # results = eval_on_all_data(kgqa_retriever, top_k_graph=top_k_graph, input_df=input_df)
    # results.to_csv("../../data/harry/modular_preds.csv",index=False)



if __name__ == "__main__":
    run_experiments()
