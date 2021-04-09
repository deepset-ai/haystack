from typing import List, Tuple, Dict, Any
import logging

from haystack import MultiLabel

from farm.evaluation.squad_evaluation import compute_f1 as calculate_f1_str
from farm.evaluation.squad_evaluation import compute_exact as calculate_em_str

logger = logging.getLogger(__name__)


class EvalRetriever:
    """
    This is a pipeline node that should be placed after a Retriever in order to assess its performance. Performance
    metrics are stored in this class and updated as each sample passes through it. To view the results of the evaluation,
    call EvalRetriever.print()
    """
    def __init__(self, debug=False, open_domain=True):
        """
        :param open_domain: When True, a document is considered correctly retrieved so long as the answer string can be found within it.
                            When False, correct retrieval is evaluated based on document_id.
        :type open_domain: bool
        :param debug: When True, a record of each sample and its evaluation will be stored in EvalRetriever.log
        :type debug: bool
        """
        self.outgoing_edges = 1
        self.init_counts()
        self.no_answer_warning = False
        self.debug = debug
        self.log = []
        self.open_domain = open_domain

    def init_counts(self):
        self.correct_retrieval_count = 0
        self.query_count = 0
        self.has_answer_count = 0
        self.has_answer_correct = 0
        self.has_answer_recall = 0
        self.no_answer_count = 0
        self.recall = 0.0

    def run(self, documents, labels: dict, **kwargs):
        """Run this node on one sample and its labels"""
        self.query_count += 1
        retriever_labels = labels["retriever"]
        # TODO retriever_labels is currently a Multilabel object but should eventually be a RetrieverLabel object
        # If this sample is impossible to answer and expects a no_answer response
        if retriever_labels.no_answer:
            self.no_answer_count += 1
            correct_retrieval = 1
            if not self.no_answer_warning:
                self.no_answer_warning = True
                logger.warning("There seem to be empty string labels in the dataset suggesting that there "
                               "are samples with is_impossible=True. "
                               "Retrieval of these samples is always treated as correct.")
        # If there are answer span annotations in the labels
        else:
            self.has_answer_count += 1
            correct_retrieval = self.is_correctly_retrieved(retriever_labels, documents)
            self.has_answer_correct += int(correct_retrieval)
            self.has_answer_recall = self.has_answer_correct / self.has_answer_count

        self.correct_retrieval_count += correct_retrieval
        self.recall = self.correct_retrieval_count / self.query_count
        if self.debug:
            self.log.append({"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs})
        return {"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs}, "output_1"

    def is_correctly_retrieved(self, retriever_labels, predictions):
        if self.open_domain:
            for label in retriever_labels.multiple_answers:
                for p in predictions:
                    if label.lower() in p.text.lower():
                        return True
            return False
        else:
            prediction_ids = [p.id for p in predictions]
            label_ids = retriever_labels.multiple_document_ids
            for l in label_ids:
                if l in prediction_ids:
                    return True
            return False

    def print(self):
        """Print the evaluation results"""
        print("Retriever")
        print("-----------------")
        if self.no_answer_count:
            print(
                f"has_answer recall: {self.has_answer_recall:.4f} ({self.has_answer_correct}/{self.has_answer_count})")
            print(
                f"no_answer recall:  1.00 ({self.no_answer_count}/{self.no_answer_count}) (no_answer samples are always treated as correctly retrieved)")
        print(f"recall: {self.recall:.4f} ({self.correct_retrieval_count} / {self.query_count})")


class EvalReader:
    """
    This is a pipeline node that should be placed after a Reader in order to assess the performance of the Reader
    individually or to assess the extractive QA performance of the whole pipeline. Performance metrics are stored in
    this class and updated as each sample passes through it. To view the results of the evaluation, call EvalReader.print()
    """

    def __init__(self, skip_incorrect_retrieval=True, open_domain=True, debug=False):
        """
        :param skip_incorrect_retrieval: When set to True, this eval will ignore the cases where the retriever returned no correct documents
        :type skip_incorrect_retrieval: bool
        :param open_domain: When True, extracted answers are evaluated purely on string similarity rather than the position of the extracted answer
        :type open_domain: bool
        :param debug: When True, a record of each sample and its evaluation will be stored in EvalReader.log
        :type debug: bool
        """
        self.outgoing_edges = 1
        self.init_counts()
        self.log = []
        self.debug = debug
        self.skip_incorrect_retrieval = skip_incorrect_retrieval
        self.open_domain = open_domain

    def init_counts(self):
        self.query_count = 0
        self.correct_retrieval_count = 0
        self.no_answer_count = 0
        self.has_answer_count = 0
        self.top_1_no_answer_count = 0
        self.top_1_em_count = 0
        self.top_k_em_count = 0
        self.top_1_f1_sum = 0
        self.top_k_f1_sum = 0
        self.top_1_no_answer = 0
        self.top_1_em = 0.0
        self.top_k_em = 0.0
        self.top_1_f1 = 0.0
        self.top_k_f1 = 0.0

    def run(self, labels, answers, **kwargs):
        """Run this node on one sample and its labels"""
        self.query_count += 1
        predictions = answers
        skip = self.skip_incorrect_retrieval and not kwargs.get("correct_retrieval")
        if predictions and not skip:
            self.correct_retrieval_count += 1
            multi_labels = labels["reader"]
            # If this sample is impossible to answer and expects a no_answer response
            if multi_labels.no_answer:
                self.no_answer_count += 1
                if predictions[0]["answer"] is None:
                    self.top_1_no_answer_count += 1
                if self.debug:
                    self.log.append({"predictions": predictions,
                                     "gold_labels": multi_labels,
                                     "top_1_no_answer": int(predictions[0] == ""),
                                     })
                self.update_no_answer_metrics()
            # If there are answer span annotations in the labels
            else:
                self.has_answer_count += 1
                predictions = [p for p in predictions if p["answer"]]
                top_1_em, top_1_f1, top_k_em, top_k_f1 = self.evaluate_extraction(multi_labels, predictions)
                if self.debug:
                    self.log.append({"predictions": predictions,
                                     "gold_labels": multi_labels,
                                     "top_k_f1": top_k_f1,
                                     "top_k_em": top_k_em
                                     })

                self.top_1_em_count += top_1_em
                self.top_1_f1_sum += top_1_f1
                self.top_k_em_count += top_k_em
                self.top_k_f1_sum += top_k_f1
                self.update_has_answer_metrics()
        return {**kwargs}, "output_1"

    def evaluate_extraction(self, gold_labels, predictions):
        if self.open_domain:
            gold_labels_list = gold_labels.multiple_answers
            predictions_str = [p["answer"] for p in predictions]
            top_1_em = calculate_em_str_multi(gold_labels_list, predictions_str[0])
            top_1_f1 = calculate_f1_str_multi(gold_labels_list, predictions_str[0])
            top_k_em = max([calculate_em_str_multi(gold_labels_list, p) for p in predictions_str])
            top_k_f1 = max([calculate_f1_str_multi(gold_labels_list, p) for p in predictions_str])
        else:
            logger.error("Closed Domain Reader Evaluation not yet implemented")
            return 0,0,0,0
        return top_1_em, top_1_f1, top_k_em, top_k_f1

    def update_has_answer_metrics(self):
        self.top_1_em = self.top_1_em_count / self.has_answer_count
        self.top_k_em = self.top_k_em_count / self.has_answer_count
        self.top_1_f1 = self.top_1_f1_sum / self.has_answer_count
        self.top_k_f1 = self.top_k_f1_sum / self.has_answer_count

    def update_no_answer_metrics(self):
        self.top_1_no_answer = self.top_1_no_answer_count / self.no_answer_count

    def print(self, mode):
        """Print the evaluation results"""
        if mode == "reader":
            print("Reader")
            print("-----------------")
            # print(f"answer in retrieved docs: {correct_retrieval}")
            print(f"has answer queries: {self.has_answer_count}")
            print(f"top 1 EM: {self.top_1_em:.4f}")
            print(f"top k EM: {self.top_k_em:.4f}")
            print(f"top 1 F1: {self.top_1_f1:.4f}")
            print(f"top k F1: {self.top_k_f1:.4f}")
            if self.no_answer_count:
                print()
                print(f"no_answer queries: {self.no_answer_count}")
                print(f"top 1 no_answer accuracy: {self.top_1_no_answer:.4f}")
        elif mode == "pipeline":
            print("Pipeline")
            print("-----------------")

            pipeline_top_1_em = (self.top_1_em_count + self.top_1_no_answer_count) / self.query_count
            pipeline_top_k_em = (self.top_k_em_count + self.no_answer_count) / self.query_count
            pipeline_top_1_f1 = (self.top_1_f1_sum + self.top_1_no_answer_count) / self.query_count
            pipeline_top_k_f1 = (self.top_k_f1_sum + self.no_answer_count) / self.query_count

            print(f"queries: {self.query_count}")
            print(f"top 1 EM: {pipeline_top_1_em:.4f}")
            print(f"top k EM: {pipeline_top_k_em:.4f}")
            print(f"top 1 F1: {pipeline_top_1_f1:.4f}")
            print(f"top k F1: {pipeline_top_k_f1:.4f}")
            if self.no_answer_count:
                print(
                    "(top k results are likely inflated since the Reader always returns a no_answer prediction in its top k)"
                )


def calculate_em_str_multi(gold_labels, prediction):
    for gold_label in gold_labels:
        result = calculate_em_str(gold_label, prediction)
        if result == 1.0:
            return 1.0
    return 0.0


def calculate_f1_str_multi(gold_labels, prediction):
    results = []
    for gold_label in gold_labels:
        result = calculate_f1_str(gold_label, prediction)
        results.append(result)
    return max(results)


def calculate_reader_metrics(metric_counts: Dict[str, float], correct_retrievals: int):
    number_of_has_answer = correct_retrievals - metric_counts["number_of_no_answer"]

    metrics = {
        "reader_top1_accuracy" : metric_counts["correct_readings_top1"] / correct_retrievals,
        "reader_top1_accuracy_has_answer" : metric_counts["correct_readings_top1_has_answer"] / number_of_has_answer,
        "reader_topk_accuracy" : metric_counts["correct_readings_topk"] / correct_retrievals,
        "reader_topk_accuracy_has_answer" : metric_counts["correct_readings_topk_has_answer"] / number_of_has_answer,
        "reader_top1_em" : metric_counts["exact_matches_top1"] / correct_retrievals,
        "reader_top1_em_has_answer" : metric_counts["exact_matches_top1_has_answer"] / number_of_has_answer,
        "reader_topk_em" : metric_counts["exact_matches_topk"] / correct_retrievals,
        "reader_topk_em_has_answer" : metric_counts["exact_matches_topk_has_answer"] / number_of_has_answer,
        "reader_top1_f1" : metric_counts["summed_f1_top1"] / correct_retrievals,
        "reader_top1_f1_has_answer" : metric_counts["summed_f1_top1_has_answer"] / number_of_has_answer,
        "reader_topk_f1" : metric_counts["summed_f1_topk"] / correct_retrievals,
        "reader_topk_f1_has_answer" : metric_counts["summed_f1_topk_has_answer"] / number_of_has_answer,
    }

    if metric_counts["number_of_no_answer"]:
        metrics["reader_top1_no_answer_accuracy"] = metric_counts["correct_no_answers_top1"] / metric_counts[
            "number_of_no_answer"]
        metrics["reader_topk_no_answer_accuracy"] = metric_counts["correct_no_answers_topk"] / metric_counts[
            "number_of_no_answer"]
    else:
        metrics["reader_top1_no_answer_accuracy"] = None  # type: ignore
        metrics["reader_topk_no_answer_accuracy"] = None  # type: ignore

    return metrics


def calculate_average_precision_and_reciprocal_rank(questions_with_docs: List[dict]):
    questions_with_correct_doc = []
    summed_avg_precision_retriever = 0.0
    summed_reciprocal_rank_retriever = 0.0

    for question in questions_with_docs:
        number_relevant_docs = len(set(question["question"].multiple_document_ids))
        found_relevant_doc = False
        relevant_docs_found = 0
        current_avg_precision = 0.0
        for doc_idx, doc in enumerate(question["docs"]):
            # check if correct doc among retrieved docs
            if doc.id in question["question"].multiple_document_ids:
                if not found_relevant_doc:
                    summed_reciprocal_rank_retriever += 1 / (doc_idx + 1)
                relevant_docs_found += 1
                found_relevant_doc = True
                current_avg_precision += relevant_docs_found / (doc_idx + 1)
                if relevant_docs_found == number_relevant_docs:
                    break
        if found_relevant_doc:
            all_relevant_docs = len(set(question["question"].multiple_document_ids))
            summed_avg_precision_retriever += current_avg_precision / all_relevant_docs

        if found_relevant_doc:
            questions_with_correct_doc.append({
                "question": question["question"],
                "docs": question["docs"]
            })

    return questions_with_correct_doc, summed_avg_precision_retriever, summed_reciprocal_rank_retriever


def eval_counts_reader(question: MultiLabel, predicted_answers: Dict[str, Any], metric_counts: Dict[str, float]):
    # Calculates evaluation metrics for one question and adds results to counter.
    # check if question is answerable
    if not question.no_answer:
        found_answer = False
        found_em = False
        best_f1 = 0
        for answer_idx, answer in enumerate(predicted_answers["answers"]):
            if answer["document_id"] in question.multiple_document_ids:
                gold_spans = [{"offset_start": question.multiple_offset_start_in_docs[i],
                               "offset_end": question.multiple_offset_start_in_docs[i] + len(question.multiple_answers[i]),
                               "doc_id": question.multiple_document_ids[i]} for i in range(len(question.multiple_answers))]  # type: ignore
                predicted_span = {"offset_start": answer["offset_start_in_doc"],
                                  "offset_end": answer["offset_end_in_doc"],
                                  "doc_id": answer["document_id"]}
                best_f1_in_gold_spans = 0
                for gold_span in gold_spans:
                    if gold_span["doc_id"] == predicted_span["doc_id"]:
                        # check if overlap between gold answer and predicted answer
                        if not found_answer:
                            metric_counts, found_answer = _count_overlap(gold_span, predicted_span, metric_counts, answer_idx)  # type: ignore

                        # check for exact match
                        if not found_em:
                            metric_counts, found_em = _count_exact_match(gold_span, predicted_span, metric_counts, answer_idx)  # type: ignore

                        # calculate f1
                        current_f1 = _calculate_f1(gold_span, predicted_span)  # type: ignore
                        if current_f1 > best_f1_in_gold_spans:
                            best_f1_in_gold_spans = current_f1
                # top-1 f1
                if answer_idx == 0:
                    metric_counts["summed_f1_top1"] += best_f1_in_gold_spans
                    metric_counts["summed_f1_top1_has_answer"] += best_f1_in_gold_spans
                if best_f1_in_gold_spans > best_f1:
                    best_f1 = best_f1_in_gold_spans

            if found_em:
                break
        # top-k answers: use best f1-score
        metric_counts["summed_f1_topk"] += best_f1
        metric_counts["summed_f1_topk_has_answer"] += best_f1

    # question not answerable
    else:
        metric_counts["number_of_no_answer"] += 1
        metric_counts = _count_no_answer(predicted_answers["answers"], metric_counts)

    return metric_counts


def eval_counts_reader_batch(pred: Dict[str, Any], metric_counts: Dict[str, float]):
    # Calculates evaluation metrics for one question and adds results to counter.

    # check if question is answerable
    if not pred["label"].no_answer:
        found_answer = False
        found_em = False
        best_f1 = 0
        for answer_idx, answer in enumerate(pred["answers"]):
            # check if correct document:
            if answer["document_id"] in pred["label"].multiple_document_ids:
                gold_spans = [{"offset_start": pred["label"].multiple_offset_start_in_docs[i],
                               "offset_end": pred["label"].multiple_offset_start_in_docs[i] + len(pred["label"].multiple_answers[i]),
                               "doc_id": pred["label"].multiple_document_ids[i]}
                              for i in range(len(pred["label"].multiple_answers))]  # type: ignore
                predicted_span = {"offset_start": answer["offset_start_in_doc"],
                                  "offset_end": answer["offset_end_in_doc"],
                                  "doc_id": answer["document_id"]}

                best_f1_in_gold_spans = 0
                for gold_span in gold_spans:
                    if gold_span["doc_id"] == predicted_span["doc_id"]:
                        # check if overlap between gold answer and predicted answer
                        if not found_answer:
                            metric_counts, found_answer = _count_overlap(
                                gold_span, predicted_span, metric_counts, answer_idx
                            )
                        # check for exact match
                        if not found_em:
                            metric_counts, found_em = _count_exact_match(
                                gold_span, predicted_span, metric_counts, answer_idx
                            )
                        # calculate f1
                        current_f1 = _calculate_f1(gold_span, predicted_span)
                        if current_f1 > best_f1_in_gold_spans:
                            best_f1_in_gold_spans = current_f1
                # top-1 f1
                if answer_idx == 0:
                    metric_counts["summed_f1_top1"] += best_f1_in_gold_spans
                    metric_counts["summed_f1_top1_has_answer"] += best_f1_in_gold_spans
                if best_f1_in_gold_spans > best_f1:
                    best_f1 = best_f1_in_gold_spans

            if found_em:
                break

        # top-k answers: use best f1-score
        metric_counts["summed_f1_topk"] += best_f1
        metric_counts["summed_f1_topk_has_answer"] += best_f1

    # question not answerable
    else:
        metric_counts["number_of_no_answer"] += 1
        metric_counts = _count_no_answer(pred["answers"], metric_counts)

    return metric_counts


def _count_overlap(
    gold_span: Dict[str, Any],
    predicted_span: Dict[str, Any],
    metric_counts: Dict[str, float],
    answer_idx: int
    ):
    # Checks if overlap between prediction and real answer.

    found_answer = False

    if (gold_span["offset_start"] <= predicted_span["offset_end"]) and \
       (predicted_span["offset_start"] <= gold_span["offset_end"]):
        # top-1 answer
        if answer_idx == 0:
            metric_counts["correct_readings_top1"] += 1
            metric_counts["correct_readings_top1_has_answer"] += 1
        # top-k answers
        metric_counts["correct_readings_topk"] += 1
        metric_counts["correct_readings_topk_has_answer"] += 1
        found_answer = True

    return metric_counts, found_answer


def _count_exact_match(
    gold_span: Dict[str, Any],
    predicted_span: Dict[str, Any],
    metric_counts: Dict[str, float],
    answer_idx: int
    ):
    # Check if exact match between prediction and real answer.
    # As evaluation needs to be framework independent, we cannot use the farm.evaluation.metrics.py functions.

    found_em = False

    if (gold_span["offset_start"] == predicted_span["offset_start"]) and \
       (gold_span["offset_end"] == predicted_span["offset_end"]):
        if metric_counts:
            # top-1 answer
            if answer_idx == 0:
                metric_counts["exact_matches_top1"] += 1
                metric_counts["exact_matches_top1_has_answer"] += 1
            # top-k answers
            metric_counts["exact_matches_topk"] += 1
            metric_counts["exact_matches_topk_has_answer"] += 1
        found_em = True

    return metric_counts, found_em


def _calculate_f1(gold_span: Dict[str, Any], predicted_span: Dict[str, Any]):
    # Calculates F1-Score for prediction based on real answer using character offsets.
    # As evaluation needs to be framework independent, we cannot use the farm.evaluation.metrics.py functions.

    pred_indices = list(range(predicted_span["offset_start"], predicted_span["offset_end"]))
    gold_indices = list(range(gold_span["offset_start"], gold_span["offset_end"]))
    n_overlap = len([x for x in pred_indices if x in gold_indices])
    if pred_indices and gold_indices and n_overlap:
        precision = n_overlap / len(pred_indices)
        recall = n_overlap / len(gold_indices)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
    else:
        return 0


def _count_no_answer(answers: List[dict], metric_counts: Dict[str, float]):
    # Checks if one of the answers is 'no answer'.

    for answer_idx, answer in enumerate(answers):
        # check if 'no answer'
        if answer["answer"] is None:
            # top-1 answer
            if answer_idx == 0:
                metric_counts["correct_no_answers_top1"] += 1
                metric_counts["correct_readings_top1"] += 1
                metric_counts["exact_matches_top1"] += 1
                metric_counts["summed_f1_top1"] += 1
            # top-k answers
            metric_counts["correct_no_answers_topk"] += 1
            metric_counts["correct_readings_topk"] += 1
            metric_counts["exact_matches_topk"] += 1
            metric_counts["summed_f1_topk"] += 1
            break

    return metric_counts
