import itertools
import json
import logging
import math
from collections import Counter

import pandas as pd

from haystack.graph_retriever.kgqa import KGQARetriever, Text2SparqlRetriever
from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType, Question
from haystack.graph_retriever.triple import Triple
from haystack.graph_retriever.utils import eval_on_all_data
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


def run_examples(kgqa_retriever: KGQARetriever, top_k_graph: int):
    result = kgqa_retriever.retrieve(question_text="What is the hair color of Hermione?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Albus Dumbledore die?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who has Blond hair?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Harry die?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the patronus of Harry?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the founder of house Gryffindor?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="How many members are in house Gryffindor?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who are the members of house Gryffindor?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the nephew of Fred Weasley?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the father of Fred Weasley?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Is Professor McGonagall in house Gryffindor?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="In which house is Professor McGonagall?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Which owner of the Marauders Map was born in Scotland?",
                                     top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the name of the daughter of Harry and Ginny?",
                                     top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="How many children does Harry Potter have?", top_k=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Does Harry Potter have a child?", top_k=top_k_graph)


def train_relation_linking(kgqa_retriever: KGQARetriever, filename: str):
    df = pd.read_csv(filename)
    df = df.head(n=10)

    number_of_sentences = 0
    document_frequency = Counter()  # each sentence corresponds to a document for idf calculation
    relation_to_token_freq = dict()
    for relation in kgqa_retriever.predicate_names:
        relation_to_token_freq[relation] = Counter()
    # "A sentence expresses relation r if it contains two co-occurring entities that are in relation r according to a knowledge base."
    for index, row in df.iterrows():
        document_text = row['document_text']
        logging.getLogger("haystack.graph_retriever.question").setLevel(logging.WARNING)
        logger.warning(f"Processing document {index + 1} out of {len(df)}")
        for sentence in kgqa_retriever.nlp(document_text).sents:
            number_of_sentences += 1
            q = Question(question_text=sentence.text)
            q.doc = kgqa_retriever.nlp(q.question_text)
            entities = q.entity_linking(alias_to_entity_and_prob=kgqa_retriever.alias_to_entity_and_prob,
                                        subject_names=kgqa_retriever.subject_names,
                                        object_names=kgqa_retriever.object_names)

            # sentences with too many entities are skipped because they dont help the training
            if len(entities) > 7:
                continue

            tokens = [token.lemma_.lower() for token in sentence]
            document_frequency.update(set(tokens))

            # for all pairs of entities, check in which relations they co-occur
            for (s, o) in itertools.permutations(entities, 2):
                # get all relations
                triples = {Triple(subject=s, predicate="?uri", object=o)}
                response = kgqa_retriever.query_executor.execute(
                    Query(question_type=QuestionType.ListQuestion, triples=triples))
                for token in tokens:
                    for relation in response:
                        if relation in kgqa_retriever.predicate_names:
                            relation_to_token_freq[relation][token] += 1

    # relation_to_token_freq[relation][token] contains term frequency
    # document_frequency[token] contains document frequency
    #
    # calculate idf for each token in vocabulary
    # calculate tf for each relation and each token
    #
    token_and_relation_to_tfidf = dict()
    for token in document_frequency:
        if document_frequency[token] < 10:
            continue
        idf = math.log(number_of_sentences / document_frequency[token], 2)
        for relation in kgqa_retriever.predicate_names:
            tf = relation_to_token_freq[relation][token]
            token_and_relation_to_tfidf[str((token, relation))] = tf * idf

    json.dump(token_and_relation_to_tfidf, open("token_and_relation_to_tfidf.json", "w"))


def run_experiments():
    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)

    # settings
    kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="xxx")
    input_df = pd.read_csv("../../data/harry/2021 03 11 Questions Apple Hackathon - original.csv").sample(n=20)

    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path="../../models/kgqa/hp_v3.2")
    top_k_graph = 1

    results = eval_on_all_data(kgqa_retriever, top_k_graph=top_k_graph, input_df=input_df)
    results.to_csv("../../data/harry/t2spqrql_preds.csv")

    # kgqa_retriever = KGQARetriever(knowledge_graph=kg, query_ranker_path="saved_models/lcquad_text_pair_classification_with_entity_labels_v2", alias_to_entity_and_prob_path="alias_to_entity_and_prob.json", token_and_relation_to_tfidf_path="token_and_relation_to_tfidf.json")
    # top_k_graph = 1
    # results = eval_on_all_data(kgqa_retriever, top_k_graph=top_k_graph, input_df=input_df)
    # results.to_csv("../../data/harry/modular_preds.csv")


    # # functionality to eval and train modular approach
    # kgqa_retriever.eval(filename="Infobox Labeling - Tabellenblatt1.csv", question_type="List", top_k_graph=top_k_graph)
    # kgqa_retriever.predictions_to_text(filename="Infobox Labeling - Tabellenblatt1.csv")
    # run_examples(kgqa_retriever=kgqa_retriever, top_k_graph=top_k_graph)
    # train_relation_linking(filename="harrypotter_docs.csv")


if __name__ == "__main__":
    run_experiments()
