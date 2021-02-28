import json
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

        self.top_relations = json.load(open("top_relations.json"))
        self.alias_to_entity_and_prob = json.load(open("alias_to_entity_and_prob.json"))
        self.entity_to_frequency = json.load(open("entity_to_frequency.json"))

    def eval(self):
        raise NotImplementedError

    def run(self, query, top_k_graph, **kwargs):
        raise NotImplementedError

    def retrieve(self, question_text: str, top_k_graph: int):
        question: Question = Question(question_text=question_text)
        entities, relations, question_type = question.analyze(nlp=self.nlp,
                                                              entity_to_frequency=self.entity_to_frequency,
                                                              alias_to_entity_and_prob=self.alias_to_entity_and_prob,
                                                              top_relations=self.top_relations)

        triples = kgqa_retriever.triple_generation(entities, relations, question_type)
        queries = kgqa_retriever.query_generation(triples, question_type)
        queries_with_scores: List[Tuple[Query, float]] = kgqa_retriever.query_ranking(queries, question.question_text)
        for ranked_query in queries_with_scores[:top_k_graph]:
            print(ranked_query)

        # todo use top_k_graph parameter, execute top k queries and return top k results
        if len(queries_with_scores) > 0:
            final_query_result = kgqa_retriever.query_executor.execute(queries_with_scores[0][0])
            print(final_query_result)
        else:
            print("No query executed. Are there any entities and relations in the question that are also in the knowledge graph?")

    def triple_generation(self, entities: Optional[Set[str]], relations: Optional[Set[str]], question_type: QuestionType) -> Set[Triple]:
        """
        Given all linked entities and relations of a question, generate triples of the form (subject, predicate, object)
        """
        # todo do not check Query.has_result if QuestionType is BooleanQuestion
        s1: Set[Triple] = set([Triple(e1, p, e2) for e1 in entities for p in relations for e2 in entities if self.query_executor.has_result({Triple(e1, p, e2)})])
        s2: Set[Triple] = set([Triple(e, p, "?uri") for e in entities for p in relations if self.query_executor.has_result({Triple(e, p, "?uri")})])
        s3: Set[Triple] = set([Triple("?uri", p, e) for e in entities for p in relations if self.query_executor.has_result({Triple("?uri", p, e)})])
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

    def query_generation(self, triples: Set[Triple], question_type: QuestionType):
        """
        Generate where_clauses by combining triples and check whether they give a result
        """
        # todo remove the check query_result_exists for boolean questions?
        queries: List[Query] = []
        for k in range(max(len(triples), 5)):
            for triple_combination in itertools.combinations(triples, k):
                if not triple_combination:
                    continue
                # check if query result exists in knowledge graph
                if self.query_executor.has_result(set(triple_combination)):
                    queries.append(Query(question_type=question_type, triples=set(triple_combination)))

        return queries

    def similarity_of_question_query(self, query, question):
        """
        Calculate the semantic similarity of a query given as a where_clause and a question
        """
        return 1.0

    def similarity_of_question_queries(self, queries, question):
        """
        Calculate the semantic similarity of each query given as a where_clause and a question
        """
        # todo we should use Tree LSTMs to calculate similarity as described in https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-020-00383-w.pdf
        # use different encoders for query and question
        # we should replace linked entities in the question with a placeholder as described in http://jens-lehmann.org/files/2018/eswc_qa_query_generation.pdf
        # train on LC-QuAD dataset
        # use text pair classification from farm and train on LC-QuAD
        basic_texts = [{"text": (question, query)} for query in queries]
        from pathlib import Path
        save_dir = Path("saved_models/text_pair_classification_model")

        # needs FARM
        # from farm.infer import Inferencer
        # model = Inferencer.load(save_dir)
        # predictions = model.inference_from_dicts(dicts=basic_texts)
        # model.close_multiprocessing_pool()
        # return [(query, prediction["probability"]) for query, prediction in zip(queries, predictions)]
        return [(query, 1.0) for query in queries]

    def query_ranking(self, queries, question):
        """
        Sort queries based on their semantic similarity with the question
        """
        # queries_with_scores = [(query,similarity_of_question_query(query,question)) for query in queries]
        queries_with_scores = self.similarity_of_question_queries(queries, question)
        queries_with_scores.sort(key=itemgetter(1), reverse=True)  # should we sort ascending or descending?
        return queries_with_scores


if __name__ == "__main__":
    kg = GraphDBKnowledgeGraph(host="34.255.232.122", username="admin", password="x-x-x")
    # load triples in the db
    # kg.import_from_ttl_file(path="triples.ttl", index="hp-test")
    # get all triples
    # results = kg.get_all_triples(index="hp-test")

    # sparql query
    # results = kg.query(query=sparql_query_str, index="hp-test")
    # print(results)

    kgqa_retriever = KGQARetriever(knowledge_graph=kg)
    kgqa_retriever.retrieve(question_text="Did Dumbledore die?", top_k_graph=1)
    #kgqa_retriever.retrieve(question_text="What is the name of the daughter of Harry and Ginny?", top_k_graph=1)
    #kgqa_retriever.retrieve(question_text="How many children does Harry Potter have?", top_k_graph=1)
    #kgqa_retriever.retrieve(question_text="Does Harry Potter have a child?", top_k_graph=1)

