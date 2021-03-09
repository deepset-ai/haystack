import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Set, Optional

import pandas as pd
import spacy
import itertools
from operator import itemgetter

from farm.infer import Inferencer

from haystack.graph_retriever.base import BaseGraphRetriever
from haystack.graph_retriever.query import Query
from haystack.graph_retriever.query_executor import QueryExecutor
from haystack.graph_retriever.question import QuestionType, Question
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

from transformers import BartForConditionalGeneration, BartTokenizer


logger = logging.getLogger(__name__)


class KGQARetriever(BaseGraphRetriever):
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.min_threshold = 2
        self.query_executor = QueryExecutor(knowledge_graph)
        # self.nlp = spacy.load('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_lg')
        # self.nlp = spacy.load('en_core_web_trf')

        self.subject_names = Counter(
            [result["s"]["value"] for result in self.knowledge_graph.get_all_subjects(index="hp-test")])
        self.predicate_names = Counter(
            [result["p"]["value"] for result in self.knowledge_graph.get_all_predicates(index="hp-test")])
        self.object_names = Counter(
            [result["o"]["value"] for result in self.knowledge_graph.get_all_objects(index="hp-test")])
        self.filter_relations_from_entities()
        logger.info(
            f"Loaded {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects.")

        # filter out subject, objects and relations that occur only once
        self.filter_infrequent_entities(min_threshold=self.min_threshold)
        logger.info(
            f"Filtered down to {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects occuring at least {self.min_threshold} times.")

        self.alias_to_entity_and_prob = json.load(open("alias_to_entity_and_prob.json"))
        self.alias_to_entity_and_prob = self.filter_existing_entities()
        # save_dir = Path("saved_models/text_pair_classification_model")
        self.save_dir = Path("saved_models/lcquad_text_pair_classification_with_entity_labels")
        self.model = Inferencer.load(self.save_dir)

    def eval(self):
        raise NotImplementedError

    def run(self, query, top_k_graph, **kwargs):
        raise NotImplementedError

    def retrieve(self, question_text: str, top_k_graph: int):
        logger.info(f"Processing question \"{question_text}\"")
        question: Question = Question(question_text=question_text)
        entities, relations, question_type = question.analyze(nlp=self.nlp,
                                                              alias_to_entity_and_prob=self.alias_to_entity_and_prob,
                                                              subject_names=self.subject_names,
                                                              predicate_names=self.predicate_names,
                                                              object_names=self.object_names)

        triples = self.triple_generation(entities, relations, question_type)
        queries = self.query_generation(triples, question_type)
        queries_with_scores: List[Tuple[Query, float]] = self.query_ranking(queries, question.question_text)
        results = []
        logger.info(f"Listing top {min(top_k_graph, len(queries_with_scores))} queries and answers (k={top_k_graph})")
        for queries_with_score in queries_with_scores[:top_k_graph]:
            result = self.query_executor.execute(queries_with_score[0])
            logger.info(f"Query: {queries_with_score[0]}")
            logger.info(f"Answer: {result}")
            results.append(result)

        if len(results) > 0:
            return results
        else:
            logger.warning(
                "No query results. Are there any entities and relations in the question that are also in the knowledge graph?")
            return None

    def filter_relations_from_entities(self):
        for predicate_name in self.predicate_names:
            if predicate_name in self.subject_names:
                self.subject_names.pop(predicate_name)
            if predicate_name in self.object_names:
                self.object_names.pop(predicate_name)

    def filter_infrequent_entities(self, min_threshold: int = 2):
        self.subject_names = {x: count for x, count in self.subject_names.items() if count >= min_threshold}
        self.predicate_names = {x: count for x, count in self.predicate_names.items() if count >= min_threshold}
        self.object_names = {x: count for x, count in self.object_names.items() if count >= min_threshold}

    def filter_existing_entities(self):
        # filter out entities that were used in anchor texts but are not in the knowledge graph
        # { k="Albus" : v=[("Albus_Dumbledore",0.9),("Albus_Potter",0.1)]}
        alias_to_entity_and_prob_filtered = dict()
        for (alias, v) in self.alias_to_entity_and_prob.items():
            filtered_entities = []
            for (entity, probability) in v:
                if Question.add_namespace_to_resource(
                        entity) in self.subject_names or Question.add_namespace_to_resource(
                        entity) in self.object_names:
                    filtered_entities.append((entity, probability))
            if filtered_entities:
                alias_to_entity_and_prob_filtered[alias] = filtered_entities
        return alias_to_entity_and_prob_filtered

    def triple_generation(self, entities: Optional[Set[str]], relations: Optional[Set[str]],
                          question_type: QuestionType) -> Set[Triple]:
        """
        Given all linked entities and relations of a question, generate triples of the form (subject, predicate, object)
        """
        # Do not check if triple exists in knowledge graph if QuestionType is BooleanQuestion
        if question_type is QuestionType.BooleanQuestion:
            s1: Set[Triple] = set([Triple(e1, p, e2) for e1 in entities for p in relations for e2 in entities])
            s2: Set[Triple] = set([Triple(e, p, "?uri") for e in entities for p in relations])
            s3: Set[Triple] = set([Triple("?uri", p, e) for e in entities for p in relations])
            s: Set[Triple] = s1.union(s2, s3)

            s_extend: Set[Triple] = set()
            for triple1 in s:
                for triple2 in s:
                    if triple1.object == "?uri" and triple2.subject == "?urix" and triple1.predicate == triple2.predicate:
                        s_extend.add(Triple("?uri", triple1.predicate, "?urix"))
                        s_extend.add(Triple("?urix", triple1.predicate, "?uri"))

            s = s.union(s_extend)
            logger.info(f"Generated {len(s)} triples")
            return s
        else:
            s1: Set[Triple] = set([Triple(e1, p, e2) for e1 in entities for p in relations for e2 in entities if
                                   self.query_executor.has_result({Triple(e1, p, e2)})])
            s2: Set[Triple] = set([Triple(e, p, "?uri") for e in entities for p in relations if
                                   self.query_executor.has_result({Triple(e, p, "?uri")})])
            s3: Set[Triple] = set([Triple("?uri", p, e) for e in entities for p in relations if
                                   self.query_executor.has_result({Triple("?uri", p, e)})])
            s: Set[Triple] = s1.union(s2, s3)

            s_extend: Set[Triple] = set()
            for triple1 in s:
                for triple2 in s:
                    if triple1.object == "?uri" and triple2.subject == "?urix" and triple1.predicate == triple2.predicate:
                        if self.query_executor.has_result({Triple("?uri", triple1.predicate, "?urix")}):
                            s_extend.add(Triple("?uri", triple1.predicate, "?urix"))
                        if self.query_executor.has_result({Triple("?urix", triple1.predicate, "?uri")}):
                            s_extend.add(Triple("?urix", triple1.predicate, "?uri"))

            s = s.union(s_extend)
            logger.info(f"Generated {len(s)} triples")
            return s

    def query_generation(self, triples: Set[Triple], question_type: QuestionType) -> List[Query]:
        """
        Generate where_clauses by combining triples and check whether they give a result
        """
        queries: List[Query] = []
        for k in range(1, min(len(triples) + 1, 4)):
            logger.info(
                f"Generating {sum(1 for _ in itertools.combinations(triples, k))} triple combinations for k={k}")
            for triple_combination in itertools.combinations(triples, k):
                if not triple_combination:
                    continue
                # check if query result exists in knowledge graph
                if question_type == QuestionType.BooleanQuestion or self.query_executor.has_result(
                        set(triple_combination)):
                    queries.append(Query(question_type=question_type, triples=set(triple_combination)))

        logger.info(f"Number of queries before pruning: {len(queries)}")
        if not question_type == QuestionType.BooleanQuestion:
            # select statement contains ?uri. therefore at least one triple in the where clause should also contain ?uri
            # and every triple should contain a subject, predicate or object that starts with "?"
            queries = [query for query in queries if query.has_variable_in_every_triple() and query.has_uri_variable()]
        logger.info(f"Number of queries after pruning: {len(queries)}")
        return queries

    def similarity_of_question_queries(self, queries, question, use_farm=True, max_ranking=5):
        """
        Calculate the semantic similarity of each query and a question
        """
        # todo we should use Tree LSTMs to calculate similarity as described in https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-020-00383-w.pdf
        # use different encoders for query and question
        # we should replace linked entities in the question with a placeholder as described in http://jens-lehmann.org/files/2018/eswc_qa_query_generation.pdf
        # train on LC-QuAD dataset
        # use text pair classification from farm and train on LC-QuAD
        if len(queries) < 2:
            # if there is no ranking needed, skip it and return score -1 because no score has been calculated
            return [(query, 1.0) for query in queries]

        if use_farm:
            if len(queries) > max_ranking:
                logger.warning(f"Ranking {len(queries)} would take too long. Ranking {max_ranking} queries instead.")
                queries = queries[:max_ranking]

            basic_texts = [{"text": (question, query.get_verbalized_sparql_query())} for query in queries]

            predictions = self.model.inference_from_dicts(dicts=basic_texts)
            #self.model.close_multiprocessing_pool()
            probabilities = []
            for prediction in predictions:
                for p in prediction["predictions"]:
                    probabilities.append(p["probability"])
            return [(query, probability) for query, probability in zip(queries, probabilities)]
        return [(query, 1.0) for query in queries]

    def query_ranking(self, queries, question):
        """
        Sort queries based on their semantic similarity with the question
        """
        logger.info(f"Ranking {len(queries)} queries")
        queries_with_scores = self.similarity_of_question_queries(queries, question)
        queries_with_scores.sort(key=itemgetter(1), reverse=True)
        return queries_with_scores

    def eval_on_extractive_qa_test_data(self, top_k_graph, filename: str):
        df = pd.read_csv(filename, sep=";")
        predictions = []
        for index, row in df.iterrows():
            if row['Category'] == "YES":
                ground_truth_answer = "True"
            elif row['Category'] == "NO":
                ground_truth_answer = "False"
            else:
                predictions.append("")
                continue
            prediction = self.retrieve(question_text=row['Question Text'], top_k_graph=top_k_graph)
            predictions.append(prediction)
            print(f"Pred: {prediction}")
            print(
                f"Label: {ground_truth_answer.replace('https://harrypotter.fandom.com/wiki/', 'https://deepset.ai/harry_potter/')}")

        df['prediction'] = predictions
        df.to_csv("20210304_harry_answers_predictions.csv", index=False)

class Text2SparqlRetriever(BaseGraphRetriever):
    def __init__(self, knowledge_graph, model_name_or_path):
        self.knowledge_graph = knowledge_graph
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, force_bos_token_to_be_generated=True)
        self.tok = BartTokenizer.from_pretrained(model_name_or_path)

    def retrieve(self, question_text: str, top_k_graph: int):
        inputs = self.tok([question_text], max_length=100,truncation=True, return_tensors='pt')
        temp = self.model.generate(inputs['input_ids'],
                                   num_beams=top_k_graph+3,
                                   max_length=100,
                                   num_return_sequences=top_k_graph+3,
                                   early_stopping=True)
        sparql_list = [self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp]
        answers = []
        for s in sparql_list:
            ans = self._query_kg(query=s)
            if ans is not None:
                answers.append(ans)
        return answers[:top_k_graph]

    def _query_kg(self, query):
        query = query.replace("}", " }")
        query = query.replace(")", " )")
        splits = query.split()
        query = ""
        for s in splits:
            if s.startswith("hp:"):
                s = s.replace("hp:", "<https://deepset.ai/harry_potter/")
                s = s + ">"
            query += s + " "
        try:
            response = self.knowledge_graph.query(query=query, index="hp-test")

            if isinstance(response, bool):
                result = response
            elif "count" in response[0]:
                result = int(response[0]["count"]["value"])
            else:
                result = []
                for x in response:
                    for k,v in x.items():
                        result.append(v["value"])
        except Exception as e:
            # print(f"Wrong query with exception: {e}")
            result = None
        return result

    def eval_on_test_data(self, top_k_graph: int, filename: str):
        # "https://deepset.ai/harry_potter/"
        # https://harrypotter.fandom.com/wiki/
        df = pd.read_csv(filename)
        predictions = []
        for index, row in df.iterrows():
            ground_truth_answer = row['answer']
            prediction = self.retrieve(question_text=row['question'], top_k_graph=top_k_graph)
            predictions.append(prediction)
            print(f"Pred: {prediction}")
            print(
                f"Label: {ground_truth_answer.replace('https://harrypotter.fandom.com/wiki/', 'https://deepset.ai/harry_potter/')}")

        df['prediction'] = predictions
        df.to_csv("predictions.csv", index=False)

    def run_examples(self, top_k_graph: int):
        # result = self.retrieve(question_text="What is the hair color of Hermione?", top_k_graph=top_k_graph)
        # result = self.retrieve(question_text="Did Albus Dumbledore die?", top_k_graph=top_k_graph)
        # result = self.query_executor.execute(Query(question_type=QuestionType.ListQuestion, triples={
        #     Triple(subject="<https://deepset.ai/harry_potter/Grawp>",
        #            predicate="<https://deepset.ai/harry_potter/hair>",
        #            object="?uri")}))
        # result = self.query_executor.execute(Query(question_type=QuestionType.ListQuestion, triples={
        #     Triple(subject="<https://deepset.ai/harry_potter/Grawp>", predicate="?uri", object="\"Brown\"")}))
        # result = self.retrieve(question_text="Who has Blond hair?", top_k_graph=top_k_graph)
        # result = self.retrieve(question_text="Did Albus Dumbledore die?", top_k_graph=top_k_graph)
        # result = self.retrieve(question_text="Did Harry die?", top_k_graph=top_k_graph)
        # result = self.retrieve(question_text="What is the patronus of Harry?", top_k_graph=top_k_graph)
        # result = self.retrieve(question_text="Who is the founder of house Gryffindor?", top_k_graph=top_k_graph)
        result = self.retrieve(question_text="How many members are in house Gryffindor?", top_k_graph=top_k_graph)
        result = self.retrieve(question_text="Who are the members of house Gryffindor?", top_k_graph=top_k_graph)
        result = self.retrieve(question_text="Who is the nephew of Fred Weasley?", top_k_graph=top_k_graph)
        result = self.retrieve(question_text="Who is the father of Fred Weasley?", top_k_graph=top_k_graph)
        result = self.retrieve(question_text="Is Professor McGonagall in house Gryffindor?",
                               top_k_graph=top_k_graph)
        result = self.retrieve(question_text="In which house is Professor McGonagall?", top_k_graph=top_k_graph)
        # result = kgqa_retriever.retrieve(question_text="Which owner of the Marauders Map was born in Scotland?", top_k_graph=top_k_graph)
        # result = kgqa_retriever.retrieve(question_text="What is the name of the daughter of Harry and Ginny?", top_k_graph=1)
        # result = kgqa_retriever.retrieve(question_text="How many children does Harry Potter have?", top_k_graph=1)
        # result = kgqa_retriever.retrieve(question_text="Does Harry Potter have a child?", top_k_graph=1)



def run():
    kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="x-x-x")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    for module in ["farm.utils", "farm.infer", "farm.modeling.prediction_head", "farm.data_handler.processor"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)
    logger.info("Starting...")

    # load triples in the db
    # kg.import_from_ttl_file(path="triples.ttl", index="hp-test")
    # return kg.query(query="ASK WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")
    # return kg.query(query="SELECT ?uri WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")

    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg, model_name_or_path='../../models/kgqa/hp_v2')

    #kgqa_retriever = KGQARetriever(knowledge_graph=kg)
    top_k_graph = 2

    # kgqa_retriever.eval_on_extractive_qa_test_data(top_k_graph=top_k_graph, filename="20210304_harry_answers.csv")
    # kgqa_retriever.eval_on_test_data(top_k_graph=top_k_graph, filename="2021_03_04_knowledge_graph_eval_questions.csv")
    kgqa_retriever.run_examples(top_k_graph=top_k_graph)


if __name__ == "__main__":
    run()

