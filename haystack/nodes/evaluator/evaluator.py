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


class EvalDocuments(BaseComponent):
    """
    This is a pipeline node that should be placed after a node that returns a List of Document, e.g., Retriever or
    Ranker, in order to assess its performance. Performance metrics are stored in this class and updated as each
    sample passes through it. To view the results of the evaluation, call EvalDocuments.print(). Note that results
    from this Node may differ from that when calling Retriever.eval() since that is a closed domain evaluation. Have
    a look at our evaluation tutorial for more info about open vs closed domain eval (
    https://haystack.deepset.ai/tutorials/evaluation).

    EvalDocuments node is deprecated and will be removed in a future version.
    Please use pipeline.eval() instead.
    """

    outgoing_edges = 1

    def __init__(self, debug: bool = False, open_domain: bool = True, top_k: int = 10):
        """
        :param open_domain: When True, a document is considered correctly retrieved so long as the answer string can be found within it.
                            When False, correct retrieval is evaluated based on document_id.
        :param debug: When True, a record of each sample and its evaluation will be stored in EvalDocuments.log
        :param top_k: calculate eval metrics for top k results, e.g., recall@k
        """
        logger.warning(
            "EvalDocuments node is deprecated and will be removed in a future version. "
            "Please use pipeline.eval() instead."
        )
        super().__init__()
        self.init_counts()
        self.no_answer_warning = False
        self.debug = debug
        self.log: List = []
        self.open_domain = open_domain
        self.top_k = top_k
        self.too_few_docs_warning = False
        self.top_k_used = 0

    def init_counts(self):
        self.correct_retrieval_count = 0
        self.query_count = 0
        self.has_answer_count = 0
        self.has_answer_correct = 0
        self.has_answer_recall = 0
        self.no_answer_count = 0
        self.recall = 0.0
        self.mean_reciprocal_rank = 0.0
        self.has_answer_mean_reciprocal_rank = 0.0
        self.reciprocal_rank_sum = 0.0
        self.has_answer_reciprocal_rank_sum = 0.0

    def run(self, documents: List[Document], labels: List[Label], top_k: Optional[int] = None):  # type: ignore
        """Run this node on one sample and its labels"""
        self.query_count += 1
        retriever_labels = get_label(labels, self.name)
        if not top_k:
            top_k = self.top_k

        if not self.top_k_used:
            self.top_k_used = top_k
        elif self.top_k_used != top_k:
            logger.warning(
                "EvalDocuments was last run with top_k_eval_documents=%s} but is "
                "being run again with top_k=%s. "
                "The evaluation counter is being reset from this point so that the evaluation "
                "metrics are interpretable.",
                self.top_k_used,
                self.top_k,
            )
            self.init_counts()

        if len(documents) < top_k and not self.too_few_docs_warning:
            logger.warning(
                "EvalDocuments is being provided less candidate documents than top_k (currently set to %s).", top_k
            )
            self.too_few_docs_warning = True

        # TODO retriever_labels is currently a Multilabel object but should eventually be a RetrieverLabel object
        # If this sample is impossible to answer and expects a no_answer response
        if retriever_labels.no_answer:
            self.no_answer_count += 1
            correct_retrieval = 1
            retrieved_reciprocal_rank = 1
            self.reciprocal_rank_sum += 1
            if not self.no_answer_warning:
                self.no_answer_warning = True
                logger.warning(
                    "There seem to be empty string labels in the dataset suggesting that there "
                    "are samples with is_impossible=True. "
                    "Retrieval of these samples is always treated as correct."
                )
        # If there are answer span annotations in the labels
        else:
            self.has_answer_count += 1
            retrieved_reciprocal_rank = self.reciprocal_rank_retrieved(retriever_labels, documents, top_k)
            self.reciprocal_rank_sum += retrieved_reciprocal_rank
            correct_retrieval = True if retrieved_reciprocal_rank > 0 else False
            self.has_answer_correct += int(correct_retrieval)
            self.has_answer_reciprocal_rank_sum += retrieved_reciprocal_rank
            self.has_answer_recall = self.has_answer_correct / self.has_answer_count
            self.has_answer_mean_reciprocal_rank = self.has_answer_reciprocal_rank_sum / self.has_answer_count

        self.correct_retrieval_count += correct_retrieval
        self.recall = self.correct_retrieval_count / self.query_count
        self.mean_reciprocal_rank = self.reciprocal_rank_sum / self.query_count

        self.top_k_used = top_k

        if self.debug:
            self.log.append(
                {
                    "documents": documents,
                    "labels": labels,
                    "correct_retrieval": correct_retrieval,
                    "retrieved_reciprocal_rank": retrieved_reciprocal_rank,
                }
            )
        return {"correct_retrieval": correct_retrieval}, "output_1"

    def run_batch(self):  # type: ignore
        raise NotImplementedError("run_batch not supported for EvalDocuments node.")

    def reciprocal_rank_retrieved(self, retriever_labels, predictions, top_k_eval_documents):
        if self.open_domain:
            for answer in retriever_labels.answers:
                for rank, p in enumerate(predictions[:top_k_eval_documents]):
                    if answer.lower() in p.content.lower():
                        return 1 / (rank + 1)
            return False
        else:
            prediction_ids = [p.id for p in predictions[:top_k_eval_documents]]
            label_ids = retriever_labels.document_ids
            for rank, p in enumerate(prediction_ids):
                if p in label_ids:
                    return 1 / (rank + 1)
            return 0

    def print(self):
        """Print the evaluation results"""
        print(self.name)
        print("-----------------")
        if self.no_answer_count:
            print(
                f"has_answer recall@{self.top_k_used}: {self.has_answer_recall:.4f} ({self.has_answer_correct}/{self.has_answer_count})"
            )
            print(
                f"no_answer recall@{self.top_k_used}:  1.00 ({self.no_answer_count}/{self.no_answer_count}) (no_answer samples are always treated as correctly retrieved)"
            )
            print(f"has_answer mean_reciprocal_rank@{self.top_k_used}: {self.has_answer_mean_reciprocal_rank:.4f}")
            print(
                f"no_answer mean_reciprocal_rank@{self.top_k_used}:  1.0000 (no_answer samples are always treated as correctly retrieved at rank 1)"
            )
        print(f"recall@{self.top_k_used}: {self.recall:.4f} ({self.correct_retrieval_count} / {self.query_count})")
        print(f"mean_reciprocal_rank@{self.top_k_used}: {self.mean_reciprocal_rank:.4f}")


class EvalAnswers(BaseComponent):
    """
    This is a pipeline node that should be placed after a Reader in order to assess the performance of the Reader
    individually or to assess the extractive QA performance of the whole pipeline. Performance metrics are stored in
    this class and updated as each sample passes through it. To view the results of the evaluation, call EvalAnswers.print().
    Note that results from this Node may differ from that when calling Reader.eval()
    since that is a closed domain evaluation. Have a look at our evaluation tutorial for more info about
    open vs closed domain eval (https://haystack.deepset.ai/tutorials/evaluation).

    EvalAnswers node is deprecated and will be removed in a future version.
    Please use pipeline.eval() instead.
    """

    outgoing_edges = 1

    def __init__(
        self,
        skip_incorrect_retrieval: bool = True,
        open_domain: bool = True,
        sas_model: Optional[str] = None,
        debug: bool = False,
    ):
        """
        :param skip_incorrect_retrieval: When set to True, this eval will ignore the cases where the retriever returned no correct documents
        :param open_domain: When True, extracted answers are evaluated purely on string similarity rather than the position of the extracted answer
        :param sas_model: Name or path of "Semantic Answer Similarity (SAS) model". When set, the model will be used to calculate similarity between predictions and labels and generate the SAS metric.
                          The SAS metric correlates better with human judgement of correct answers as it does not rely on string overlaps.
                          Example: Prediction = "30%", Label = "thirty percent", EM and F1 would be overly pessimistic with both being 0, while SAS paints a more realistic picture.
                          More info in the paper: https://arxiv.org/abs/2108.06130
                          Models:
                          - You can use Bi Encoders (sentence transformers) or cross encoders trained on Semantic Textual Similarity (STS) data.
                            Not all cross encoders can be used because of different return types.
                            If you use custom cross encoders please make sure they work with sentence_transformers.CrossEncoder class
                          - Good default for multiple languages: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                          - Large, powerful, but slow model for English only: "cross-encoder/stsb-roberta-large"
                          - Large model for German only: "deepset/gbert-large-sts"
        :param debug: When True, a record of each sample and its evaluation will be stored in EvalAnswers.log
        """
        logger.warning(
            "EvalAnswers node is deprecated and will be removed in a future version. "
            "Please use pipeline.eval() instead."
        )
        super().__init__()
        self.log: List = []
        self.debug = debug
        self.skip_incorrect_retrieval = skip_incorrect_retrieval
        self.open_domain = open_domain
        self.sas_model = sas_model
        self.init_counts()

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
        if self.sas_model is not None:
            self.top_1_sas_sum = 0
            self.top_k_sas_sum = 0
            self.top_1_sas = 0.0
            self.top_k_sas = 0.0

    def run(self, labels: List[Label], answers: List[Answer], correct_retrieval: bool):  # type: ignore
        """Run this node on one sample and its labels"""
        self.query_count += 1
        predictions: List[Answer] = answers
        skip = self.skip_incorrect_retrieval and not correct_retrieval
        if predictions and not skip:
            self.correct_retrieval_count += 1
            multi_labels = get_label(labels, self.name)
            # If this sample is impossible to answer and expects a no_answer response
            if multi_labels.no_answer:
                self.no_answer_count += 1
                if predictions[0].answer is None:
                    self.top_1_no_answer_count += 1
                if self.debug:
                    self.log.append(
                        {
                            "predictions": predictions,
                            "gold_labels": multi_labels,
                            "top_1_no_answer": int(predictions[0].answer is None),
                        }
                    )
                self.update_no_answer_metrics()
            # If there are answer span annotations in the labels
            else:
                self.has_answer_count += 1
                predictions_str: List[str] = [p.answer if p.answer else "" for p in predictions]
                top_1_em, top_1_f1, top_k_em, top_k_f1 = self.evaluate_extraction(multi_labels.answers, predictions_str)

                # Compute Semantic Answer Similarity if model is supplied
                if self.sas_model is not None:
                    # sas works on batches, so we pack the labels into a list of lists, and unpack the return values as well
                    top_1_sas, top_k_sas, _ = semantic_answer_similarity(
                        predictions=[predictions_str],
                        gold_labels=[multi_labels.answers],
                        sas_model_name_or_path=self.sas_model,
                    )
                    self.top_1_sas_sum += top_1_sas[0]
                    self.top_k_sas_sum += top_k_sas[0]

                if self.debug:
                    self.log.append(
                        {
                            "predictions": predictions,
                            "gold_labels": multi_labels,
                            "top_k_f1": top_k_f1,
                            "top_k_em": top_k_em,
                        }
                    )
                    if self.sas_model:
                        self.log[-1].update({"top_k_sas": top_k_sas})

                self.top_1_em_count += top_1_em
                self.top_1_f1_sum += top_1_f1
                self.top_k_em_count += top_k_em
                self.top_k_f1_sum += top_k_f1
                self.update_has_answer_metrics()
        return {}, "output_1"

    def run_batch(self):  # type: ignore
        raise NotImplementedError("run_batch not supported for EvalAnswers node.")

    def evaluate_extraction(self, gold_labels: List[str], predictions: List[str]):
        if self.open_domain:
            top_1_em = calculate_em_str_multi(gold_labels, predictions[0])
            top_1_f1 = calculate_f1_str_multi(gold_labels, predictions[0])
            top_k_em = max(calculate_em_str_multi(gold_labels, p) for p in predictions)
            top_k_f1 = max(calculate_f1_str_multi(gold_labels, p) for p in predictions)
        else:
            logger.error(
                "Closed Domain Reader Evaluation not yet implemented for Pipelines. Use Reader.eval() instead."
            )
            return 0, 0, 0, 0
        return top_1_em, top_1_f1, top_k_em, top_k_f1

    def update_has_answer_metrics(self):
        self.top_1_em = self.top_1_em_count / self.has_answer_count
        self.top_k_em = self.top_k_em_count / self.has_answer_count
        self.top_1_f1 = self.top_1_f1_sum / self.has_answer_count
        self.top_k_f1 = self.top_k_f1_sum / self.has_answer_count
        if self.sas_model is not None:
            self.top_1_sas = self.top_1_sas_sum / self.has_answer_count
            self.top_k_sas = self.top_k_sas_sum / self.has_answer_count

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
            if self.sas_model is not None:
                print(f"top 1 SAS: {self.top_1_sas:.4f}")
                print(f"top k SAS: {self.top_k_sas:.4f}")
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
            if self.sas_model is not None:
                pipeline_top_1_sas = (self.top_1_sas_sum + self.top_1_no_answer_count) / self.query_count
                pipeline_top_k_sas = (self.top_k_sas_sum + self.no_answer_count) / self.query_count
                print(f"top 1 SAS: {pipeline_top_1_sas:.4f}")
                print(f"top k SAS: {pipeline_top_k_sas:.4f}")
            if self.no_answer_count:
                print(
                    "(top k results are likely inflated since the Reader always returns a no_answer prediction in its top k)"
                )


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
