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

logger = logging.getLogger(__name__)


class KGQARetriever(BaseGraphRetriever):
    def __init__(
        self,
        knowledge_graph,
        query_ranker_path,
        alias_to_entity_and_prob_path,
    ):
        self.knowledge_graph: GraphDBKnowledgeGraph = knowledge_graph
        self.min_threshold: int = 2
        self.query_ranker: QueryRanker = QueryRanker(query_ranker_path)
        self.query_executor: QueryExecutor = QueryExecutor(knowledge_graph)
        # self.nlp = spacy.load('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_lg')
        # self.nlp = spacy.load('en_core_web_trf')

        logger.info("Loading triples from knowledge graph...")
        self.subject_names = Counter(
            [result["s"]["value"] for result in self.knowledge_graph.get_all_subjects(index="hp-test")])
        self.predicate_names = Counter(
            [result["p"]["value"] for result in self.knowledge_graph.get_all_predicates(index="hp-test")])
        self.object_names = Counter(
            [result["o"]["value"] for result in self.knowledge_graph.get_all_objects(index="hp-test")])
        self.filter_relations_from_entities()
        logger.info(
            f"Loaded {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects.")

        # filter out subjects, objects and relations that occur only once
        self.filter_infrequent_entities_and_relations(min_threshold=self.min_threshold)
        logger.info(
            f"Filtered down to {len(self.subject_names)} subjects, {len(self.predicate_names)} predicates and {len(self.object_names)} objects occuring at least {self.min_threshold} times.")

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

            answer = [str(answer_item).strip().lower() for answer_item in re.split(",|\ \ ", str(answer))]
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
        df = pd.read_csv(filename, sep="\t")
        df = df[df['Type'] == question_type]
        predictions_for_all_queries = []
        number_of_correct_predictions = 0
        correct_predictions = []
        for index, row in df.iterrows():
            predictions_for_query = self.retrieve(question_text=row['Question'], top_k_graph=top_k_graph)
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

    def run(self, query, top_k_graph, **kwargs):
        return self.retrieve(question_text=query, top_k_graph=top_k_graph)

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
        queries_with_scores = self.query_ranker.query_ranking(queries=queries, question=question.question_text, top_k_graph=top_k_graph)
        results = []
        logger.info(f"Listing top {min(top_k_graph, len(queries_with_scores))} queries and answers (k={top_k_graph})")
        for queries_with_score in queries_with_scores[:top_k_graph]:
            result = self.query_executor.execute(queries_with_score[0])
            logger.info(f"Score: {queries_with_score[1]} Query: {queries_with_score[0]}")
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

    def filter_infrequent_entities_and_relations(self, min_threshold: int = 2):
        self.subject_names = {x: count for x, count in self.subject_names.items() if count >= min_threshold}
        # filter top 100 relations
        self.predicate_names = {x: count for x, count in self.predicate_names.items() if count >= min_threshold and (x, count) in self.predicate_names.most_common(100)}
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

    def eval_on_extractive_qa_test_data(self, top_k_graph: int, filename: str):
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

    def distant_supervision(self):
        # import pandas as pd
        # import spacy
        # nlp = spacy.load('en_core_web_lg')
        df = pd.read_csv("harrypotter_docs.csv")
        df = df.head(n=1)



        document_frequency = Counter() # each sentence corresponds to a document for idf calculation
        relation_to_token = dict()
        for relation in self.predicate_names:
            relation_to_token[relation] = Counter()
        # works -> (job, 1000)
        # "A sentence expresses relation r if it contains two co-occurring entities that are in relation r according to a knowledge base."
        for index, row in df.iterrows():
            document_text = row['document_text']
            logging.getLogger("haystack.graph_retriever.question").setLevel(logging.WARNING)
            logger.warning(f"Processing document {index+1} out of {len(df)}")
            for sentence in self.nlp(document_text).sents:
                entities = [token.text for token in sentence]
                q = Question(question_text=sentence.text)
                q.doc = self.nlp(q.question_text)
                entities = q.entity_linking(alias_to_entity_and_prob=self.alias_to_entity_and_prob,
                        subject_names=self.subject_names,
                        predicate_names=self.predicate_names,
                        object_names=self.object_names,
                        fuzzy=False)

                # sentences with too many entities are skipped because they dont help the training
                if len(entities) > 5:
                    continue

                tokens = [token.text.lower() for token in sentence]
                document_frequency.update(set(tokens))

                # for all pairs of entities, check in which relations they co-occur
                for (s, o) in itertools.permutations(entities, 2):
                    # get all relations
                    triples = {Triple(subject=s, predicate="?uri", object=o)}
                    response = self.query_executor.execute(Query(question_type=QuestionType.ListQuestion, triples=triples))
                    for token in tokens:
                        for relation in response:
                            if relation in self.predicate_names:
                                relation_to_token[relation][token] += 1

        print("completed")

        # relation_to_token[relation][token] contains term frequency
        # document_frequency[token] contains document frequency
        #
        # calculate idf for each word in vocabulary
        # calculate tf for each relation and each word
        #
        relation_to_best_token = dict() #or vice versa
        for relation in self.predicate_names:
            best_tf_idf = 0
            for token in document_frequency:
                if best_tf_idf < relation_to_token[relation][token]/document_frequency[token] and document_frequency[token] > 10:
                    best_tf_idf = relation_to_token[relation][token] / document_frequency[token]
                    relation_to_best_token[relation] = token

        # goal: dictionary: token -> most likely relation and the corresponding tf_idf value
        token_to_relation = dict()
        for token in document_frequency:
            if document_frequency[token] <= 10:
                continue
            best_tf = 0
            for relation in self.predicate_names:
                if best_tf < relation_to_token[relation][token]:
                    best_tf = relation_to_token[relation][token]
                    token_to_relation[token] = (relation, best_tf/document_frequency[token])

        json.dump(relation_to_best_token, open("relation_to_best_token.json", "w"))
        json.dump(token_to_relation, open("token_to_relation.json", "w"))
        json.dump(document_frequency, open("document_frequency.json", "w"))
