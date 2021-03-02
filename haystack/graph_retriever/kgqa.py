import json
import logging
from collections import Counter
from typing import List, Tuple, Set, Optional

import spacy
import itertools
from operator import itemgetter

from haystack.graph_retriever.base import BaseGraphRetriever
from haystack.graph_retriever.query import Query
from haystack.graph_retriever.query_executor import QueryExecutor
from haystack.graph_retriever.question import QuestionType, Question
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph


class KGQARetriever(BaseGraphRetriever):
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.query_executor = QueryExecutor(knowledge_graph)
        # self.nlp = spacy.load('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_lg')
        # self.nlp = spacy.load('en_core_web_trf')

        self.subject_names = Counter([result["s"]["value"] for result in kg.get_all_subjects(index="hp-test")])
        self.predicate_names = Counter([result["p"]["value"] for result in kg.get_all_predicates(index="hp-test")])
        self.object_names = Counter([result["o"]["value"] for result in kg.get_all_objects(index="hp-test")])

        #self.top_relations = self.predicate_names.keys()
        self.alias_to_entity_and_prob = json.load(open("alias_to_entity_and_prob.json"))
        self.alias_to_entity_and_prob = self.filter_existing_entities()

    def eval(self):
        raise NotImplementedError

    def run(self, query, top_k_graph, **kwargs):
        raise NotImplementedError

    def retrieve(self, question_text: str, top_k_graph: int):
        question: Question = Question(question_text=question_text)
        entities, relations, question_type = question.analyze(nlp=self.nlp,
                                                              alias_to_entity_and_prob=self.alias_to_entity_and_prob,
                                                              subject_names=self.subject_names,
                                                              predicate_names=self.predicate_names,
                                                              object_names=self.object_names)

        triples = kgqa_retriever.triple_generation(entities, relations, question_type)
        queries = kgqa_retriever.query_generation(triples, question_type)
        queries_with_scores: List[Tuple[Query, float]] = kgqa_retriever.query_ranking(queries, question.question_text)
        results = []
        logger.info(f"Listing top {top_k_graph} queries and answers for question \"{question.question_text}\"")
        logger.info(f"Linked entities: {question.entities}")
        logger.info(f"Linked relations: {question.relations}")
        for queries_with_score in queries_with_scores[:top_k_graph]:
            result = kgqa_retriever.query_executor.execute(queries_with_score[0])
            logger.info(f"Query: {queries_with_score[0]}")
            logger.info(f"Answer: {result}")
            results.append(result)

        if len(results) > 0:
            return results
        else:
            logger.warning("No query results. Are there any entities and relations in the question that are also in the knowledge graph?")
            return None

    def filter_existing_entities(self):
        # filter out entities that were used in anchor texts but are not in the knowledge graph
        # { k="Albus" : v=[("Albus_Dumbledore",0.9),("Albus_Potter",0.1)]}
        alias_to_entity_and_prob_filtered = dict()
        # {k:v for (k,v) in self.alias_to_entity_and_prob.items() if v in self.subject_names or v in self.object_names}
        for (alias, v) in self.alias_to_entity_and_prob.items():
            filtered_entities = []
            for (entity, probability) in v:
                if f"https://deepset.ai/harry_potter/{entity.replace(' ', '_').replace('.', '_')}" in self.subject_names or f"https://deepset.ai/harry_potter/{entity.replace(' ', '_').replace('.', '_')}" in self.object_names:
                    filtered_entities.append((entity, probability))
            if filtered_entities:
                alias_to_entity_and_prob_filtered[alias] = filtered_entities
        return alias_to_entity_and_prob_filtered

    def triple_generation(self, entities: Optional[Set[str]], relations: Optional[Set[str]], question_type: QuestionType) -> Set[Triple]:
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
            return s

    def query_generation(self, triples: Set[Triple], question_type: QuestionType) -> List[Query]:
        """
        Generate where_clauses by combining triples and check whether they give a result
        """
        queries: List[Query] = []
        for k in range(max(len(triples), 5)):
            for triple_combination in itertools.combinations(triples, k):
                if not triple_combination:
                    continue
                # check if query result exists in knowledge graph
                if question_type == QuestionType.BooleanQuestion or self.query_executor.has_result(set(triple_combination)):
                    queries.append(Query(question_type=question_type, triples=set(triple_combination)))

        return queries

    def similarity_of_question_queries(self, queries, question, use_farm=False):
        """
        Calculate the semantic similarity of each query and a question
        """
        # todo we should use Tree LSTMs to calculate similarity as described in https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-020-00383-w.pdf
        # use different encoders for query and question
        # we should replace linked entities in the question with a placeholder as described in http://jens-lehmann.org/files/2018/eswc_qa_query_generation.pdf
        # train on LC-QuAD dataset
        # use text pair classification from farm and train on LC-QuAD
        if use_farm:
            basic_texts = [{"text": (question, query.get_sparql_query())} for query in queries]
            from pathlib import Path
            save_dir = Path("saved_models/text_pair_classification_model")

            # needs FARM
            from farm.infer import Inferencer
            model = Inferencer.load(save_dir)
            predictions = model.inference_from_dicts(dicts=basic_texts)
            model.close_multiprocessing_pool()
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
        queries_with_scores = self.similarity_of_question_queries(queries, question)
        queries_with_scores.sort(key=itemgetter(1), reverse=True)  # should we sort ascending or descending?
        return queries_with_scores


if __name__ == "__main__":
    kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="x-x-x")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # load triples in the db
    # kg.import_from_ttl_file(path="triples.ttl", index="hp-test")
    # return kg.query(query="ASK WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")
    # return kg.query(query="SELECT ?uri WHERE { <https://deepset.ai/harry_potter/Albus_Dumbledore> <https://deepset.ai/harry_potter/died> ?uri }", index="hp-test")

    kgqa_retriever = KGQARetriever(knowledge_graph=kg)
    top_k_graph = 1
    result = kgqa_retriever.retrieve(question_text="Did Albus Dumbledore die?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Did Harry die?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="What is the patronus of Harry?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Who is the founder of house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="How many members are in house Gryffindor?", top_k_graph=top_k_graph)
    result = kgqa_retriever.retrieve(question_text="Which owner of the Marauders Map was born in Scotland?", top_k_graph=top_k_graph)

    #kgqa_retriever.retrieve(question_text="What is the name of the daughter of Harry and Ginny?", top_k_graph=1)
    #kgqa_retriever.retrieve(question_text="How many children does Harry Potter have?", top_k_graph=1)
    #kgqa_retriever.retrieve(question_text="Does Harry Potter have a child?", top_k_graph=1)
