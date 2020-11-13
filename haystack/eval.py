from typing import List, Tuple, Dict, Any

from haystack import MultiLabel


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