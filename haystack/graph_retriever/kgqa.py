import json
import logging
import re
from collections import Counter
from typing import List, Set, Optional

import pandas as pd
import spacy
import itertools


from haystack.graph_retriever.base import BaseGraphRetriever
from haystack.graph_retriever.query import Query
from haystack.graph_retriever.query_executor import QueryExecutor
from haystack.graph_retriever.query_ranker import QueryRanker
from haystack.graph_retriever.question import QuestionType, Question
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class KGQARetriever(BaseGraphRetriever):
    def __init__(
        self,
        knowledge_graph: GraphDBKnowledgeGraph,
        query_ranker_path: str,
        alias_to_entity_and_prob_path: str,
        token_and_relation_to_tfidf_path: str,
        top_k: int = 10
    ):
        self.knowledge_graph: GraphDBKnowledgeGraph = knowledge_graph
        self.min_freq_entities_and_relations: int = 2
        self.query_ranker: QueryRanker = QueryRanker(query_ranker_path, max_ranking=5)
        self.top_k = top_k
        self.query_executor: QueryExecutor = QueryExecutor(knowledge_graph)
        self.nlp = spacy.load('en_core_web_lg')
        self.relation_tfidf = json.load(open(token_and_relation_to_tfidf_path))

        logger.info("Loading triples from knowledge graph...")
        self.subject_names = Counter(
            [result["s"]["value"] for result in self.knowledge_graph.get_all_subjects()])
        self.predicate_names = Counter(
            [result["p"]["value"] for result in self.knowledge_graph.get_all_predicates()])
        self.object_names = Counter(
            [result["o"]["value"] for result in self.knowledge_graph.get_all_objects()])
        self.filter_relations_from_entities()
        logger.info(
            f"Loaded {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects.")

        # filter out subjects, objects and relations that occur only once
        self.filter_infrequent_entities_and_relations(min_threshold=self.min_freq_entities_and_relations)
        logger.info(
            f"Filtered down to {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects occuring at least {self.min_freq_entities_and_relations} times.")

        self.alias_to_entity_and_prob = json.load(open(alias_to_entity_and_prob_path))
        self.alias_to_entity_and_prob = self.filter_existing_entities()

    def compare_answers(self, answer, prediction, question_type: str):
        if question_type == "List":
            if isinstance(prediction, int):
                # assumed wrong question type
                return False
            # split multiple entities
            # strip whitespaces, lowercase, and remove namespace
            # convert answers to sets so that the order of the items does not matter
            answer = [str(answer_item).strip().lower() for answer_item in re.split(",|\ \ |\n", str(answer))]
            answer = {answer_item.replace('https://harrypotter.fandom.com/wiki/', "").replace("_", " ").replace("-", " ") for answer_item in answer}

            prediction = [str(prediction_item).strip().lower() for prediction_item in prediction]
            prediction = {prediction_item.replace('https://deepset.ai/harry_potter/', "").replace("_", " ") for prediction_item in prediction}

        elif question_type == "Boolean":
            answer = bool(answer)
            prediction = bool(prediction)
        elif question_type == "Count":
            answer = int(answer)
            prediction = int(prediction)

        return answer == prediction

    def eval(self, filename, question_type, top_k_graph):
        """
        Calculate top_k accuracy given a tsv file with question and answer columns and store predictions
        Do this evaluation for one chosen type of questions (List, Boolean, Count)
        """
        df = pd.read_csv(filename)
        df = df[df['Type'] == question_type]
        predictions_for_all_queries = []
        number_of_correct_predictions = 0
        correct_predictions = []
        for index, row in df.iterrows():
            predictions_for_query = self.retrieve(question_text=row['Question'], top_k=top_k_graph)
            if not predictions_for_query:
                predictions_for_all_queries.append(None)
                correct_predictions.append(0)
                continue
            top_k_contain_correct_answer = False
            for prediction in predictions_for_query:
                top_k_contain_correct_answer = self.compare_answers(row['Answer'], prediction, row['Type'])
                if top_k_contain_correct_answer:
                    number_of_correct_predictions += 1
                    correct_predictions.append(1)
                    logger.info("Correct answer.")
                    break
            if not top_k_contain_correct_answer:
                logger.info(f"Wrong answer(s). Expected {row['Answer']}")
                correct_predictions.append(0)
            predictions_for_all_queries.append(predictions_for_query)

        logger.info(f"{number_of_correct_predictions} correct answers out of {len(df)} for k={top_k_graph}")
        df['prediction'] = predictions_for_all_queries
        df['correct'] = correct_predictions
        df.to_csv("predictions.csv", index=False)

    def run(self, query, top_k_graph: Optional[int] = None, **kwargs):
        answers = self.retrieve(question_text=query, top_k=top_k_graph)

        results = {"query": query,
                   "answers": answers,
                   **kwargs}
        return results, "output_1"

    def retrieve(self, question_text: str, top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        logger.info(f"Processing question \"{question_text}\"")
        question: Question = Question(question_text=question_text)
        entities, relations, question_type = question.analyze(nlp=self.nlp,
                                                              alias_to_entity_and_prob=self.alias_to_entity_and_prob,
                                                              subject_names=self.subject_names,
                                                              predicate_names=self.predicate_names,
                                                              object_names=self.object_names,
                                                              relation_tfidf=self.relation_tfidf)

        triples = self.triple_generation(entities, relations, question_type)
        queries = self.query_generation(triples, question_type)
        queries_with_scores = self.query_ranker.query_ranking(queries=queries, question=question.question_text, top_k_graph=top_k)
        results = []
        logger.info(f"Listing top {min(top_k, len(queries_with_scores))} queries and answers (k={top_k})")
        for queries_with_score in queries_with_scores[:top_k]:
            result = self.query_executor.execute(queries_with_score[0])
            logger.info(f"Score: {queries_with_score[1]} Query: {queries_with_score[0]}")
            logger.info(f"Answer: {result}")
            results.append(result)

        if len(results) == 0:
            logger.debug(
                "No query results. Are there any entities and relations in the question that are also in the knowledge graph?")
            return {"answer": "", "meta": {"model": "GraphRetriever"}}

        print([self.format_result(result) for result in results])
        return [self.format_result(result) for result in results]
        #return results

    def format_result(self, result):
        """
        Generate formatted dictionary output with text answer and additional info
        """
        text_answer = self.prediction_to_text(result)
        meta = {"model": "GraphRetriever"}
        if True:
            meta["urls"] = str(self.prediction_to_urls(result))
        return {"answer": str(text_answer), "meta": meta}

    def filter_relations_from_entities(self):
        for predicate_name in self.predicate_names:
            if predicate_name in self.subject_names:
                self.subject_names.pop(predicate_name)
            if predicate_name in self.object_names:
                self.object_names.pop(predicate_name)

    def filter_infrequent_entities_and_relations(self, min_threshold: int = 2):
        self.subject_names = {x: count for x, count in self.subject_names.items() if count >= min_threshold}
        # filter top 100 relations
        self.predicate_names = {x: count for x, count in self.predicate_names.items() if count >= min_threshold and (x, count) in self.predicate_names.most_common(100)}
        self.object_names = {x: count for x, count in self.object_names.items() if count >= min_threshold}

    def filter_existing_entities(self):
        # filter out entities that were used in anchor texts but are not in the knowledge graph
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
            # we exclude triples where the subject is also the object although that would be valid in rare cases
            # we also exclude triples where the subject or object has the same name as the relation although that would be valid in rare cases
            s1: Set[Triple] = set([Triple(e1, p, e2) for e1 in entities for p in relations for e2 in entities if e1 != e2 and e1.lower() != p.lower() and e2.lower() != p.lower()])
            s2: Set[Triple] = set([Triple(e, p, "?uri") for e in entities for p in relations if e.lower() != p.lower()])
            s3: Set[Triple] = set([Triple("?uri", p, e) for e in entities for p in relations if e.lower() != p.lower()])
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

    def prediction_to_urls(self, prediction):
        if isinstance(prediction, bool) or isinstance(prediction, int):
            return None
        elif isinstance(prediction, list):
            return [entity.replace("https://deepset.ai/harry_potter/", "https://harrypotter.fandom.com/wiki/") for entity in prediction if entity.startswith("https://deepset.ai/harry_potter/")]
        elif not prediction.startswith("https://"):
            return None
        else:
            #split list and for each entity get url
            entities = re.split(",|\ \ |\n", prediction)
            return [entity.replace("https://deepset.ai/harry_potter/", "https://harrypotter.fandom.com/wiki/") for entity in entities if entity.startswith("https://deepset.ai/harry_potter/")]

    def predictions_to_text(self, filename):
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            print("\""+self.prediction_to_text(row['Answer']).lower()+"\"")

    def prediction_to_text(self, prediction):
        if isinstance(prediction, bool) or isinstance(prediction, int):
            return prediction
        elif isinstance(prediction, list):
            return "\n".join([self.entity_to_text(entity.strip()) for entity in prediction])
        elif not prediction.startswith("https://"):
            return prediction
        else:
            #split list and for each entity get text representation
            entities = re.split(",|\ \ |\n", prediction)
            return "\n".join([self.entity_to_text(entity.strip()) for entity in entities])

    def entity_to_text(self, entity):
        if entity.startswith("https://deepset.ai/harry_potter/"):
            triples = {Triple(subject=f"<{entity}>", predicate="<https://deepset.ai/harry_potter/name>", object="?uri")}
            response = self.query_executor.execute(Query(question_type=QuestionType.ListQuestion, triples=triples))
            if len(response) > 0:
                entity = response[0]
            entity = entity.replace("https://deepset.ai/harry_potter/", "").replace("_", " ").replace("-", " ")
        elif entity.startswith("https://harrypotter.fandom.com/wiki/"):
            entity = entity.replace("https://harrypotter.fandom.com/wiki/", "").replace("_", " ").replace("-", " ")
        return entity


class Text2SparqlRetriever(KGQARetriever):
    def __init__(self, knowledge_graph, model_name_or_path):
        self.knowledge_graph = knowledge_graph
        self.query_executor: QueryExecutor = QueryExecutor(knowledge_graph)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, force_bos_token_to_be_generated=True)
        self.tok = BartTokenizer.from_pretrained(model_name_or_path)

    def retrieve(self, question_text: str, top_k_graph: int):
        inputs = self.tok([question_text], max_length=100, truncation=True, return_tensors='pt')
        temp = self.model.generate(inputs['input_ids'],
                                   num_beams=top_k_graph+3,
                                   max_length=100,
                                   num_return_sequences=top_k_graph+3,
                                   early_stopping=True)
        sparql_list = [self.tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in temp]
        answers = []
        for s in sparql_list:
            ans = self._query_kg(query=s)
            if len(ans) > 0:
                answers.append(ans)

        # if there are no answers we still want to return something
        if len(answers) == 0:
            answers.append("")
        results = answers[:top_k_graph]
        results = [self.format_result(result) for result in results]
        return results

    def run(self, query, top_k_graph: Optional[int] = None, **kwargs):
        answers = self.retrieve(question_text=query, top_k_graph=top_k_graph)

        results = {"query": query,
                   "answers": answers,
                   **kwargs}
        return results, "output_1"

    def format_result(self, result):
        """
        Generate formatted dictionary output with text answer and additional info
        """
        text_answer = self.prediction_to_text(result)
        meta = {"model": "Question2SparqlRetriever"}
        if True:
            meta["urls"] = str(self.prediction_to_urls(result))
        return {"answer": str(text_answer), "meta": meta}

    def _query_kg(self, query):
        # Bring generated query into Harry Potter KG form
        query = query.replace("}", " }")
        query = query.replace(")", " )")
        splits = query.split()
        query = ""
        for s in splits:
            if s.startswith("hp:"):
                s = s.replace("hp:", "<https://deepset.ai/harry_potter/")
                s = s + ">"
            query += s + " "

        # query KG
        try:
            response = self.knowledge_graph.query(query=query)
        except Exception:
            return ""

        # unpack different answer styles
        if isinstance(response, list):
            if len(response) == 0:
                return ""
        if isinstance(response, bool):
            result = str(response)
        elif "count" in response[0]:
            result = str(int(response[0]["count"]["value"]))
        else:
            result = []
            for x in response:
                for k,v in x.items():
                    result.append(v["value"])
        return result

