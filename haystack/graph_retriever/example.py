import json
import logging
from collections import Counter
from typing import List, Set, Optional

import pandas as pd
import spacy
import itertools

from haystack.graph_retriever import KGQARetriever
from haystack.graph_retriever.base import BaseGraphRetriever
from haystack.graph_retriever.query import Query
from haystack.graph_retriever.query_executor import QueryExecutor
from haystack.graph_retriever.query_ranker import QueryRanker
from haystack.graph_retriever.question import QuestionType, Question
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


def run_examples(kgqa_retriever: KGQARetriever, top_k_graph: int):
    result = kgqa_retriever.retrieve(question_text="What is the hair color of Hermione?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Albus Dumbledore die?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who has Blond hair?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Harry die?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the patronus of Harry?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the founder of house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="How many members are in house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who are the members of house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the nephew of Fred Weasley?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the father of Fred Weasley?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Is Professor McGonagall in house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="In which house is Professor McGonagall?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Which owner of the Marauders Map was born in Scotland?",
                           top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the name of the daughter of Harry and Ginny?",
                           top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="How many children does Harry Potter have?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Does Harry Potter have a child?", top_k_graph=top_k_graph)


def run_experiments():
    kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="x-x-x")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)

    kgqa_retriever = KGQARetriever(knowledge_graph=kg)
    top_k_graph = 1

    # kgqa_retriever.eval_on_extractive_qa_test_data(top_k_graph=top_k_graph, filename="20210304_harry_answers.csv")
    # kgqa_retriever.run_examples(top_k_graph=top_k_graph)
    # result = kgqa_retriever.retrieve(question_text="Who founded Dumbledore's Army?", top_k_graph=top_k_graph)
    # result = kgqa_retriever.retrieve(question_text="What colour are Lorcan d'Eath's hair?", top_k_graph=top_k_graph)
    #result = kgqa_retriever.retrieve(question_text="", top_k_graph=top_k_graph)

    #result = kgqa_retriever.retrieve(question_text="Who was a seeker of the Ivorian National Quidditch team?", top_k_graph=top_k_graph)
    #result = kgqa_retriever.retrieve(question_text="What is Edith Nesbit's blood status?", top_k_graph=top_k_graph)
    #result = kgqa_retriever.retrieve(question_text="Who founded Dumbledore's Army?", top_k_graph=top_k_graph)
    kgqa_retriever.eval(filename="Infobox Labeling - Tabellenblatt3.tsv", question_type="List", top_k_graph=top_k_graph)
    run_examples(kgqa_retriever=kgqa_retriever, top_k_graph=top_k_graph)

    # todo
    #  correct handling of 's in Dumbledore's Army vs. Ronald Weasley's nicknames


if __name__ == "__main__":
    run_experiments()
