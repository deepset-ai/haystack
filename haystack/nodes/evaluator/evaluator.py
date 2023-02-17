from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from haystack.schema import MultiLabel, Label, Document, Answer
from haystack.nodes.base import BaseComponent

from haystack.modeling.evaluation.squad import compute_f1 as calculate_f1_str
from haystack.modeling.evaluation.squad import compute_exact as calculate_em_str


logger = logging.getLogger(__name__)


def get_label(labels, node_id):
    if type(labels) in [Label, MultiLabel]:
        ret = labels
    # If labels is a dict, then fetch the value using node_id (e.g. "EvalRetriever") as the key
    else:
        ret = labels[node_id]
    return ret


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
    if len(results) > 0:
        return max(results)
    else:
        return 0.0


def semantic_answer_similarity(
    predictions: List[List[str]],
    gold_labels: List[List[str]],
    sas_model_name_or_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 32,
    use_gpu: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Computes Transformer-based similarity of predicted answer to gold labels to derive a more meaningful metric than EM or F1.
    Returns per QA pair a) the similarity of the most likely prediction (top 1) to all available gold labels
                        b) the highest similarity of all predictions to gold labels
                        c) a matrix consisting of the similarities of all the predictions compared to all gold labels

    :param predictions: Predicted answers as list of multiple preds per question
    :param gold_labels: Labels as list of multiple possible answers per question
    :param sas_model_name_or_path: SentenceTransformers semantic textual similarity model, should be path or string
                                     pointing to downloadable models.
    :param batch_size: Number of prediction label pairs to encode at once.
    :param use_gpu: Whether to use a GPU or the CPU for calculating semantic answer similarity.
                    Falls back to CPU if no GPU is available.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :return: top_1_sas, top_k_sas, pred_label_matrix
    """
    assert len(predictions) == len(gold_labels)

    config = AutoConfig.from_pretrained(sas_model_name_or_path, use_auth_token=use_auth_token)
    cross_encoder_used = False
    if config.architectures is not None:
        cross_encoder_used = any(arch.endswith("ForSequenceClassification") for arch in config.architectures)

    device = None if use_gpu else "cpu"

    # Compute similarities
    top_1_sas = []
    top_k_sas = []
    pred_label_matrix = []
    lengths: List[Tuple[int, int]] = []

    # Based on Modelstring we can load either Bi-Encoders or Cross Encoders.
    # Similarity computation changes for both approaches
    if cross_encoder_used:
        model = CrossEncoder(
            sas_model_name_or_path,
            device=device,
            tokenizer_args={"use_auth_token": use_auth_token},
            automodel_args={"use_auth_token": use_auth_token},
        )
        grid = []
        for preds, labels in zip(predictions, gold_labels):
            for p in preds:
                for l in labels:
                    grid.append((p, l))
            lengths.append((len(preds), len(labels)))
        scores = model.predict(grid, batch_size=batch_size)

        current_position = 0
        for len_p, len_l in lengths:
            scores_window = scores[current_position : current_position + len_p * len_l]
            # Per predicted doc there are len_l entries comparing it to all len_l labels.
            # So to only consider the first doc we have to take the first len_l entries
            top_1_sas.append(np.max(scores_window[:len_l]))
            top_k_sas.append(np.max(scores_window))
            pred_label_matrix.append(scores_window.reshape(len_p, len_l).tolist())
            current_position += len_p * len_l
    else:
        # For Bi-encoders we can flatten predictions and labels into one list
        model = SentenceTransformer(sas_model_name_or_path, device=device, use_auth_token=use_auth_token)
        all_texts: List[str] = []
        for p, l in zip(predictions, gold_labels):  # type: ignore
            # TODO potentially exclude (near) exact matches from computations
            all_texts.extend(p)
            all_texts.extend(l)
            lengths.append((len(p), len(l)))
        # then compute embeddings
        embeddings = model.encode(all_texts, batch_size=batch_size)

        # then select which embeddings will be used for similarity computations
        current_position = 0
        for len_p, len_l in lengths:
            pred_embeddings = embeddings[current_position : current_position + len_p, :]
            current_position += len_p
            label_embeddings = embeddings[current_position : current_position + len_l, :]
            current_position += len_l
            sims = cosine_similarity(pred_embeddings, label_embeddings)
            top_1_sas.append(np.max(sims[0, :]))
            top_k_sas.append(np.max(sims))
            pred_label_matrix.append(sims.tolist())

    return top_1_sas, top_k_sas, pred_label_matrix


def _count_overlap(
    gold_span: Dict[str, Any], predicted_span: Dict[str, Any], metric_counts: Dict[str, float], answer_idx: int
):
    # Checks if overlap between prediction and real answer.

    found_answer = False

    if (gold_span["offset_start"] <= predicted_span["offset_end"]) and (
        predicted_span["offset_start"] <= gold_span["offset_end"]
    ):
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
    gold_span: Dict[str, Any], predicted_span: Dict[str, Any], metric_counts: Dict[str, float], answer_idx: int
):
    # Check if exact match between prediction and real answer.
    # As evaluation needs to be framework independent, we cannot use the farm.evaluation.metrics.py functions.

    found_em = False

    if (gold_span["offset_start"] == predicted_span["offset_start"]) and (
        gold_span["offset_end"] == predicted_span["offset_end"]
    ):
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
