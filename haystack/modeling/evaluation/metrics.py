from typing import Callable, Dict, List

import logging
from functools import reduce

import numpy as np
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report as token_classification_report
from sklearn.metrics import matthews_corrcoef, f1_score, mean_squared_error, r2_score, classification_report

from haystack.modeling.model.prediction_head import PredictionHead
from haystack.modeling.utils import flatten_list


logger = logging.getLogger(__name__)


registered_metrics = {}
registered_reports = {}


def register_metrics(name: str, implementation: Callable):
    registered_metrics[name] = implementation


def register_report(name: str, implementation: Callable):
    """
    Register a custom reporting function to be used during eval.

    This can be useful:
    - if you want to overwrite a report for an existing output type of prediction head (e.g. "per_token")
    - if you have a new type of prediction head and want to add a custom report for it

    :param name: This must match the `ph_output_type` attribute of the PredictionHead for which the report should be used.
                 (e.g. TokenPredictionHead => `per_token`, YourCustomHead => `some_new_type`).
    :param implementation: Function to be executed. It must take lists of `y_true` and `y_pred` as input and return a
                           printable object (e.g. string or dict).
                           See sklearns.metrics.classification_report for an example.
    :type implementation: function
    """
    registered_reports[name] = implementation


def simple_accuracy(preds, labels):
    # works also with nested lists of different lengths (needed for masked LM task)
    if type(preds) == type(labels) == list:
        preds = np.array(list(flatten_list(preds)))
        labels = np.array(list(flatten_list(labels)))
    assert type(preds) == type(labels) == np.ndarray
    correct = preds == labels
    return {"acc": correct.mean()}


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": acc["acc"], "f1": f1, "acc_and_f1": (acc["acc"] + f1) / 2}


def f1_macro(preds, labels):
    return {"f1_macro": f1_score(y_true=labels, y_pred=preds, average="macro")}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson": pearson_corr, "spearman": spearman_corr, "corr": (pearson_corr + spearman_corr) / 2}


def compute_metrics(metric: str, preds, labels):
    """
    Calculate the named metric values for the list of predictions vs list of labels.

    :param metric: The name of a predefined metric; a function that takes a prediction list and a label
        list and returns a dict from metric names to values, or recursively a list of metrics.
        Predefined metrics are: mcc, acc, acc_f1, pear_spear, seq_f1, f1_macro, squad, mse, r2,
        top_n_accuracy, text_similarity_metric.
    :param preds: list of predictions
    :param labels: list of target labels
    :return: a dictionary mapping metric names to values.
    """
    FUNCTION_FOR_METRIC = {
        "mcc": lambda preds, labels: {"mcc": matthews_corrcoef(labels, preds)},
        "acc": simple_accuracy,
        "acc_f1": acc_and_f1,
        "pear_spear": pearson_and_spearman,
        "f1_macro": f1_macro,
        "squad": squad,
        "mse": lambda preds, labels: {"mse": mean_squared_error(preds, labels)},
        "r2": lambda preds, labels: {"r2": r2_score(preds, labels)},
        "top_n_accuracy": lambda preds, labels: {"top_n_accuracy": top_n_accuracy(preds, labels)},
        "text_similarity_metric": text_similarity_metric,
    }
    assert len(preds) == len(labels)
    if metric in FUNCTION_FOR_METRIC.keys():
        return FUNCTION_FOR_METRIC[metric](preds, labels)
    elif isinstance(metric, list):
        ret = {}
        for m in metric:
            ret.update(compute_metrics(m, preds, labels))
        return ret
    elif metric in registered_metrics:
        metric_func = registered_metrics[metric]
        return metric_func(preds, labels)
    else:
        raise KeyError(metric)


def compute_report_metrics(head: PredictionHead, preds, labels):
    if head.ph_output_type in registered_reports:
        report_fn = registered_reports[head.ph_output_type]
    elif head.ph_output_type == "per_token":
        report_fn = token_classification_report
    elif head.ph_output_type == "per_sequence":
        report_fn = classification_report
    elif head.ph_output_type == "per_token_squad":
        report_fn = lambda *args, **kwargs: "Not Implemented"  # pylint: disable=unnecessary-lambda-assignment
    elif head.ph_output_type == "per_sequence_continuous":
        report_fn = r2_score
    else:
        raise AttributeError(
            f"No report function for head.ph_output_type '{head.ph_output_type}'. "
            f"You can register a custom one via register_report(name='{head.ph_output_type}', implementation=<your_report_function>"
        )

    # CHANGE PARAMETERS, not all report_fn accept digits
    if head.ph_output_type in ["per_sequence"]:
        # supply labels as all possible combination because if ground truth labels do not cover
        # all values in label_list (maybe dev set is small), the report will break
        if head.model_type == "text_similarity":
            labels = reduce(lambda x, y: x + list(y.astype("long")), labels, [])
            preds = reduce(lambda x, y: x + [0] * y[0] + [1] + [0] * (len(y) - y[0] - 1), preds, [])  # type: ignore
            all_possible_labels = list(range(len(head.label_list)))
        else:
            all_possible_labels = head.label_list
        return report_fn(labels, preds, digits=4, labels=all_possible_labels, target_names=head.label_list)
    else:
        return report_fn(labels, preds)


def squad_EM(preds, labels):
    """
    Count how often the pair of first predicted start and end index exactly matches one of the labels
    """
    n_docs = len(preds)
    n_correct = 0
    for (pred, label) in zip(preds, labels):
        qa_candidate = pred[0][0]
        pred_start = qa_candidate.offset_answer_start
        pred_end = qa_candidate.offset_answer_end
        curr_labels = label
        if (pred_start, pred_end) in curr_labels:
            n_correct += 1
    return n_correct / n_docs if n_docs else 0


def top_n_EM(preds, labels):
    """
    Count how often the pair of predicted start and end index exactly matches one of the labels
    """
    n_docs = len(preds)
    n_correct = 0
    for (pred, label) in zip(preds, labels):
        qa_candidates = pred[0]
        for qa_candidate in qa_candidates:
            pred_start = qa_candidate.offset_answer_start
            pred_end = qa_candidate.offset_answer_end
            curr_labels = label
            if (pred_start, pred_end) in curr_labels:
                n_correct += 1
                break
    return n_correct / n_docs if n_docs else 0


def squad_EM_start(preds, labels):
    """
    Count how often the predicted start index exactly matches the start index given by one of the labels
    """
    n_docs = len(preds)
    n_correct = 0
    for (pred, label) in zip(preds, labels):
        qa_candidate = pred[0][0]
        pred_start = qa_candidate.offset_answer_start
        curr_labels = label
        curr_labels_start = [curr_label[0] for curr_label in curr_labels]
        if pred_start in curr_labels_start:
            n_correct += 1
    return n_correct / n_docs if n_docs else 0


def squad_f1(preds, labels):
    """Calculates the f1 score (token overlap) of the first prediction"""
    f1_scores = []
    n_docs = len(preds)
    for i in range(n_docs):
        best_pred = preds[i][0]
        best_f1 = max(squad_f1_single(best_pred, label) for label in labels[i])
        f1_scores.append(best_f1)
    return np.mean(f1_scores)


def top_n_f1(preds, labels):
    f1_scores = []
    for pred, label in zip(preds, labels):
        pred_candidates = pred[0]
        best_f1 = max(
            squad_f1_single([pred_candidate], label_candidate)
            for label_candidate in label
            for pred_candidate in pred_candidates
        )
        f1_scores.append(best_f1)
    return np.mean(f1_scores)


def squad_f1_single(pred, label, pred_idx: int = 0):
    label_start, label_end = label
    span = pred[pred_idx]
    pred_start = span.offset_answer_start
    pred_end = span.offset_answer_end

    if (pred_start + pred_end == 0) or (label_start + label_end == 0):
        if pred_start == label_start:
            return 1.0
        else:
            return 0.0
    pred_span = list(range(pred_start, pred_end + 1))
    label_span = list(range(label_start, label_end + 1))
    n_overlap = len([x for x in pred_span if x in label_span])
    if n_overlap == 0:
        return 0.0
    precision = n_overlap / len(pred_span)
    recall = n_overlap / len(label_span)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def confidence(preds):
    conf = 0
    for pred in preds:
        conf += pred[0][0].confidence
    return conf / len(preds) if len(preds) else 0


def metrics_per_bin(preds, labels, num_bins: int = 10):
    pred_bins = [[] for _ in range(num_bins)]  # type: List
    label_bins = [[] for _ in range(num_bins)]  # type: List
    count_per_bin = [0] * num_bins
    for (pred, label) in zip(preds, labels):
        current_score = pred[0][0].confidence
        if current_score >= 1.0:
            current_score = 0.9999
        pred_bins[int(current_score * num_bins)].append(pred)
        label_bins[int(current_score * num_bins)].append(label)
        count_per_bin[int(current_score * num_bins)] += 1

    em_per_bin = [0] * num_bins
    confidence_per_bin = [0] * num_bins
    for i in range(num_bins):
        em_per_bin[i] = squad_EM_start(preds=pred_bins[i], labels=label_bins[i])
        confidence_per_bin[i] = confidence(preds=pred_bins[i])
    return em_per_bin, confidence_per_bin, count_per_bin


def squad_base(preds, labels):
    em = squad_EM(preds=preds, labels=labels)
    f1 = squad_f1(preds=preds, labels=labels)
    top_acc = top_n_accuracy(preds=preds, labels=labels)
    return {"EM": em, "f1": f1, "top_n_accuracy": top_acc}


def squad(preds, labels):
    """
    This method calculates squad evaluation metrics a) overall, b) for questions with text answer and c) for questions with no answer
    """
    # TODO change check for no_answer questions from using (start,end)==(-1,-1) to is_impossible flag in QAInput. This needs to be done for labels though. Not for predictions.
    overall_results = squad_base(preds, labels)

    preds_answer = [pred for (pred, label) in zip(preds, labels) if (-1, -1) not in label]
    labels_answer = [label for label in labels if (-1, -1) not in label]
    answer_results = squad_base(preds_answer, labels_answer)
    top_n_em_answer = top_n_EM(preds_answer, labels_answer)
    top_n_f1_answer = top_n_f1(preds_answer, labels_answer)

    preds_no_answer = [pred for (pred, label) in zip(preds, labels) if (-1, -1) in label]
    labels_no_answer = [label for label in labels if (-1, -1) in label]
    no_answer_results = squad_base(preds_no_answer, labels_no_answer)

    return {
        "EM": overall_results["EM"],  # this is top_1 only
        "f1": overall_results["f1"],  # this is top_1 only
        "top_n_accuracy": overall_results["top_n_accuracy"],
        "EM_text_answer": answer_results["EM"],  # this is top_1 only
        "f1_text_answer": answer_results["f1"],  # this is top_1 only
        "top_n_accuracy_text_answer": answer_results["top_n_accuracy"],
        "top_n_EM_text_answer": top_n_em_answer,
        "top_n_f1_text_answer": top_n_f1_answer,
        "Total_text_answer": len(preds_answer),
        "EM_no_answer": no_answer_results["EM"],  # this is top_1 only
        "f1_no_answer": no_answer_results["f1"],  # this is top_1 only
        "top_n_accuracy_no_answer": no_answer_results["top_n_accuracy"],
        "Total_no_answer": len(preds_no_answer),
    }


def top_n_accuracy(preds, labels):
    """
    This method calculates the percentage of documents for which the model makes top n accurate predictions.
    The definition of top n accurate a top n accurate prediction is as follows:
    For any given question document pair, there can be multiple predictions from the model and multiple labels.
    If any of those predictions overlap at all with any of the labels, those predictions are considered to be top n accurate.
    """
    answer_in_top_n = []
    n_questions = len(preds)
    for i in range(n_questions):
        f1_score = 0
        current_preds = preds[i][0]
        for idx, pred in enumerate(current_preds):
            f1_score = max(squad_f1_single(current_preds, label, pred_idx=idx) for label in labels[i])
            if f1_score:
                break
        if f1_score:
            answer_in_top_n.append(1)
        else:
            answer_in_top_n.append(0)
    return np.mean(answer_in_top_n)


def text_similarity_acc_and_f1(preds, labels):
    """
    Returns accuracy and F1 scores for top-1(highest) ranked sequence(context/passage) for each sample/query

    :param preds: list of numpy arrays of dimension n1 x n2 containing n2 predicted ranks for n1 sequences/queries
    :type preds: List of numpy array containing similarity scores for each sequence in batch
    :param labels: list of arrays of dimension n1 x n2 where each array contains n2 labels(0/1) indicating whether the sequence/passage is a positive(1) passage or hard_negative(0) passage
    :type labels: List of list containing values(0/1)

    :return: predicted ranks of passages for each query
    """
    top_1_pred = reduce(lambda x, y: x + [0] * y[0] + [1] + [0] * (len(y) - y[0] - 1), preds, [])
    labels = reduce(lambda x, y: x + list(y.astype("long")), labels, [])
    res = acc_and_f1(top_1_pred, labels)
    return res


def text_similarity_avg_ranks(preds, labels) -> float:
    """
    Calculates average predicted rank of positive sequence(context/passage) for each sample/query

    :param preds: list of numpy arrays of dimension n1 x n2 containing n2 predicted ranks for n1 sequences/queries
    :type preds: List of numpy array containing similarity scores for each sequence in batch
    :param labels: list of arrays of dimension n1 x n2 where each array contains n2 labels(0/1) indicating whether the sequence/passage is a positive(1) passage or hard_negative(0) passage
    :type labels: List of list containing values(0/1)

    :return: average predicted ranks of positive sequence/passage for each sample/query
    """
    positive_idx_per_question = list(reduce(lambda x, y: x + list((y == 1).nonzero()[0]), labels, []))  # type: ignore
    rank = 0
    for i, idx in enumerate(positive_idx_per_question):
        # aggregate the rank of the known gold passage in the sorted results for each question
        gold_idx = (preds[i] == idx).nonzero()[0]
        rank += gold_idx.item()
    return float(rank / len(preds))


def text_similarity_metric(preds, labels) -> Dict[str, float]:
    """
    Returns accuracy, F1 scores and average rank scores for text similarity task

    :param preds: list of numpy arrays of dimension n1 x n2 containing n2 predicted ranks for n1 sequences/queries
    :type preds: List of numpy array containing similarity scores for each sequence in batch
    :param labels: list of arrays of dimension n1 x n2 where each array contains n2 labels(0/1) indicating whether the sequence/passage is a positive(1) passage or hard_negative(0) passage
    :type labels: List of list containing values(0/1)

    :return: metrics(accuracy, F1, average rank) for text similarity task
    """
    scores = text_similarity_acc_and_f1(preds, labels)
    scores["average_rank"] = text_similarity_avg_ranks(preds, labels)
    return scores
