import logging
from operator import itemgetter
from pathlib import Path
from typing import Set, List, Union, Tuple

from farm.infer import Inferencer

from haystack.graph_retriever.query import Query
from haystack.graph_retriever.question import QuestionType
from haystack.graph_retriever.triple import Triple
from haystack.knowledge_graph.graphdb import GraphDBKnowledgeGraph

logger = logging.getLogger(__name__)


class QueryRanker:

    def __init__(self, filename: str):
        self.save_dir = Path(filename)
        self.model = Inferencer.load(self.save_dir)

    def query_ranking(self, queries, question, top_k_graph) -> List[Tuple[Query, float]]:
        """
        Sort queries based on their semantic similarity with the question
        """
        logger.info(f"Ranking {len(queries)} queries")
        queries_with_scores = self.similarity_of_question_queries(queries=queries, question=question, use_farm=True, top_k_graph=top_k_graph, max_ranking=5)
        queries_with_scores.sort(key=itemgetter(1), reverse=True)
        return queries_with_scores

    def similarity_of_question_queries(self, queries, question, use_farm, top_k_graph, max_ranking) -> List[Tuple[Query,float]]:
        """
        Calculate the semantic similarity of each query and a question
        Current approach uses text pair classification from FARM and is pre-trained on LC-QuAD
        Alternative approach could use Tree LSTMs to calculate similarity as described in https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-020-00383-w.pdf
        """
        max_ranking = max(top_k_graph, max_ranking)
        if len(queries) < 2:
            # if there is no ranking needed, skip it and return score -1 because no score has been calculated
            return [(query, 1.0) for query in queries]

        if use_farm:
            if len(queries) > max_ranking:
                logger.warning(f"Ranking {len(queries)} queries would take some time. Ranking {max_ranking} queries instead.")
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
