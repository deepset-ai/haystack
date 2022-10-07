import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from haystack.schema import Answer
from haystack.nodes.evaluator.evaluator import semantic_answer_similarity
from rest_api.schema import QuestionAnswerPair, PipelineHyperParams
from rest_api.utils import get_pipelines
from haystack import Pipeline
from document_indexing.s3_storage import S3Storage
from experiments.wandb_logger import WandBLogger


class PipelineEvaluation:
    @staticmethod
    def ranked_answers(answers: List[Answer]) -> List[str]:
        return [a.answer for a in sorted(answers, key=lambda a: a.score) if a.context]

    @staticmethod
    def get_n_worst_best_examples(
        questions: List[str], scores: List[float], retrieved_questions: List[List[str]], n: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sorted_scores_index = np.argsort(scores)
        index_worst = sorted_scores_index[:n]
        index_best = sorted_scores_index[-n:]
        k = len(retrieved_questions[0])
        worst = pd.DataFrame(
            data=[retrieved_questions[i] for i in index_worst], columns=[f"rank{i}" for i in range(1, k + 1)]
        )
        worst["question"] = [questions[i] for i in index_worst]
        best = pd.DataFrame(
            data=[retrieved_questions[i] for i in index_best], columns=[f"rank{i}" for i in range(1, k + 1)]
        )
        best["question"] = [questions[i] for i in index_best]
        return worst, best

    def get_performance_metrics(
        self, question_answer_pairs: List[QuestionAnswerPair], pipeline_hyper_params: PipelineHyperParams
    ):
        query_pipeline: Pipeline = get_pipelines(pipeline_hyper_params).get("query_pipeline", None)
        true_questions = []
        true_answers = []
        faq_answers = []
        extractive_answers = []

        # for each question answering pair, query the pipeline
        for q_and_a_pair in question_answer_pairs:
            true_answers.append([q_and_a_pair.answer])
            true_questions.append(q_and_a_pair.question)

            # faq
            result_faq = query_pipeline.run(query=q_and_a_pair.question, params={"CustomClassifier": {"index": "faq"}})
            faq_answers.append(self.ranked_answers(result_faq["answers"]))

            # extractive
            result_extractive = query_pipeline.run(
                query=q_and_a_pair.question, params={"CustomClassifier": {"index": "extractive"}}
            )
            extractive_answers.append(self.ranked_answers(result_extractive["answers"]))

        # for the faq, you can compute accuracy, and sas
        top1_acc = self.top_k_accuracy(true_answers, faq_answers, k=1)
        topk_acc = self.top_k_accuracy(true_answers, faq_answers, k=pipeline_hyper_params.top_k)
        rec_rank = self.reciprocal_rank(true_answers, faq_answers)
        # also get some example of good and bad cases
        n_worst_reciprocal_rank, n_best_reciprocal_rank = self.get_n_worst_best_examples(
            true_questions, rec_rank, faq_answers
        )
        faq_top1_sas, faq_topk_sas = self.mean_semantic_answer_similarity(true_answers, true_answers)

        # for extractive, we can only do sas at the moment
        extr_top1_sas, extr_topk_sas = self.mean_semantic_answer_similarity(true_answers, extractive_answers)

        metrics = {
            "faq_mean_accuracy_top_1": sum(top1_acc) / len(top1_acc),
            "faq_mean_accuracy_top_k": sum(topk_acc) / len(topk_acc),
            "faq_mean_reciprocal_rank": sum(rec_rank) / len(rec_rank),
            "faq_mean_semantic_answer_similarity_top_1": sum(faq_top1_sas) / len(faq_top1_sas),
            "faq_mean_semantic_answer_similarity_top_k": sum(faq_topk_sas) / len(faq_topk_sas),
            "extractive_mean_semantic_answer_similarity_top_1": sum(extr_top1_sas) / len(extr_top1_sas),
            "extractive_mean_semantic_answer_similarity_top_k": sum(extr_topk_sas) / len(extr_topk_sas),
        }

        # also get some example of good and bad cases
        n_worst_sas, n_best_sas = self.get_n_worst_best_examples(true_questions, extr_topk_sas, extractive_answers)

        examples = {
            "faq_worst_reciprocal_rank": n_worst_reciprocal_rank,
            "faq_best_reciprocal_rank": n_best_reciprocal_rank,
            "extractive_worst_sas_top_k": n_worst_sas,
            "extractive_best_sas_top_k": n_best_sas,
        }

        return metrics, examples

    @staticmethod
    def top_k_accuracy(true_answers: List[List[str]], retrieved_answers: List[List[str]], k) -> List[float]:
        accuracy = []
        for true_answer_list, retrieved_answer_list in zip(true_answers, retrieved_answers):
            retrieved_answer_list = retrieved_answer_list[:k]
            scores = [
                int(answer in true_answer) for answer in retrieved_answer_list for true_answer in true_answer_list
            ]
            accuracy.append(max(scores))
        return accuracy

    @staticmethod
    def reciprocal_rank(true_answers: List[List[str]], retrieved_answers: List[List[str]]) -> List[float]:
        """
        reciprocal rank = 1/position of first relevant answer
        """

        reciprocal_ranks = []

        for true_answer_list, retrieved_answer_list in zip(true_answers, retrieved_answers):
            scores = [
                int(answer in true_answer) / (1 + rank)
                for (rank, answer) in enumerate(retrieved_answer_list)
                for true_answer in true_answer_list
            ]
            reciprocal_ranks.append(max(scores))

        return reciprocal_ranks

    @staticmethod
    def mean_semantic_answer_similarity(
        true_answers: List[List[str]], retrieved_answers: List[List[str]]
    ) -> (float, float):
        top_1_sas, top_k_sas, _ = semantic_answer_similarity(
            predictions=retrieved_answers,
            gold_labels=true_answers,
            sas_model_name_or_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            batch_size=32,
            use_gpu=False,
            use_auth_token=False,  # only needed of downloading private models from huggingface
        )
        return top_1_sas, top_k_sas


if __name__ == "__main__":

    pipeline_hyper_params = PipelineHyperParams(**json.load(open("configuration.json", "r")))
    evaluator = PipelineEvaluation()
    storage = S3Storage()
    q_and_a_pairs = storage.load_qa_pairs("monopoly")
    #
    metrics, example_tables = evaluator.get_performance_metrics(q_and_a_pairs, pipeline_hyper_params)
    logger = WandBLogger(project_name="monopoly", job_name="evaluate")
    for title, table in example_tables.items():
        logger.log_table(table, title)

    logger.log_metrics(metrics)
    logger.log_metrics(pipeline_hyper_params.dict())
    logger.commit_logs()
