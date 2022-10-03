from typing import List, Tuple

import numpy as np

from haystack.schema import Answer
from haystack.nodes.evaluator.evaluator import semantic_answer_similarity
from rest_api.schema import QuestionAnswerPair
from rest_api.utils import get_pipelines
from haystack import Pipeline
from document_indexing.s3_storage import S3Storage


class FaQEvaluation:
    def ranked_questions(self, answers: List[Answer]) -> List[str]:
        """
        the matched question is the context property of the answer
        """
        return [a.context for a in sorted(answers, key=lambda a: a.score) if a.context]

    def get_n_worst_best_examples(
        self, scores: List[float], retrieved_questions: List[List[str]], n: int = 5
    ) -> Tuple[List[str], List[str]]:
        index_worst = np.nargmin(scores, n=n)
        index_best = np.nanargmin(scores, n=n)
        return retrieved_questions[index_worst], retrieved_questions[index_best]

    def get_performance_metrics(self, question_answer_pairs: List[QuestionAnswerPair]):
        query_pipeline: Pipeline = get_pipelines().get("query_pipeline", None)
        true_questions = []
        retrieved_questions = []
        k = []

        # for each question answering pair, query the pipeline
        for q_and_a_pair in question_answer_pairs[:3]:
            result = query_pipeline.run(query=q_and_a_pair.question)
            true_questions.append([q_and_a_pair.question, q_and_a_pair.alternative_question])
            ranked_questions = self.ranked_questions(result["answers"])
            k.append(len(ranked_questions))
            retrieved_questions.append(ranked_questions)

        top1_acc = self.top_k_accuracy(true_questions, retrieved_questions, k=1)
        topk_acc = self.top_k_accuracy(true_questions, retrieved_questions, k=max(k))
        rec_rank = self.reciprocal_rank(true_questions, retrieved_questions)

        top1_sas, topk_sas = self.mean_semantic_answer_similarity(true_questions, retrieved_questions)
        return {
            "mean_accuracy_top_1": sum(top1_acc) / len(top1_acc),
            "mean_accuracy_top_k": sum(topk_acc) / len(topk_acc),
            "mean_reciprocal_rank": sum(rec_rank) / len(rec_rank),
            "mean_semantic_answer_similarity_top_1": top1_sas,
            "mean_semantic_answer_similarity_top_k": topk_sas,
            "k": max([k]),
        }

    @staticmethod
    def top_k_accuracy(true_questions: List[List[str]], retrieved_questions: List[List[str]], k) -> List[float]:
        accuracy = []
        for true_question_list, retrieved_question_list in zip(true_questions, retrieved_questions):
            retrieved_question_list = retrieved_question_list[:k]
            scores = [
                int(true_question == question)
                for question in retrieved_question_list
                for true_question in true_question_list
            ]
            accuracy.append(max(scores))
        return accuracy

    @staticmethod
    def reciprocal_rank(true_questions: List[List[str]], retrieved_questions: List[List[str]]) -> List[float]:
        """
        reciprocal rank = 1/position of first relevant answer
        """

        reciprocal_ranks = []

        for true_question_list, retrieved_question_list in zip(true_questions, retrieved_questions):
            scores = [
                int(true_question == question) / (1 + rank)
                for (rank, question) in enumerate(retrieved_question_list)
                for true_question in true_question_list
            ]
            reciprocal_ranks.append(max(scores))

        return reciprocal_ranks

    @staticmethod
    def mean_semantic_answer_similarity(
        true_questions: List[List[str]], retrieved_questions: List[List[str]]
    ) -> (float, float):
        top_1_sas, top_k_sas, _ = semantic_answer_similarity(
            predictions=retrieved_questions,
            gold_labels=true_questions,
            sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            batch_size=32,
            use_gpu=False,
            use_auth_token=False,  # only needed of downloading private models from huggingface
        )
        return top_1_sas, top_k_sas


if __name__ == "__main__":
    ev = FaQEvaluation()
    storage = S3Storage()
    q_and_a_pairs = storage.load_qa_pairs("monopoly")
    ev.get_performance_metrics(q_and_a_pairs)
