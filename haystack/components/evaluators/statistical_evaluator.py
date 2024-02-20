import collections
import itertools
from enum import Enum
from typing import Any, Dict, List, Union

from numpy import array as np_array
from numpy import mean as np_mean

from haystack import default_from_dict, default_to_dict
from haystack.core.component import component


class StatisticalMetric(Enum):
    """
    Metrics supported by the StatisticalEvaluator.
    """

    F1 = "f1"
    EM = "exact_match"
    RECALL_SINGLE_HIT = "recall_single_hit"
    RECALL_MULTI_HIT = "recall_multi_hit"
    MRR = "mean_reciprocal_rank"

    @classmethod
    def from_str(cls, metric: str) -> "StatisticalMetric":
        map = {e.value: e for e in StatisticalMetric}
        metric_ = map.get(metric)
        if metric_ is None:
            raise ValueError(f"Unknown statistical metric '{metric}'")
        return metric_


@component
class StatisticalEvaluator:
    """
    StatisticalEvaluator is a component that evaluates the performance of a model based on statistical metrics.
    It's usually used in QA and Retrieval Augmented Generation (RAG) pipelines to evaluate the quality of the generated answers.

    The supported metrics are:
    - F1: Measures word overlap between predictions and labels.
    - Exact Match: Measures the proportion of cases where prediction is identical to the expected label.
    """

    def __init__(self, metric: Union[str, StatisticalMetric]):
        """
        Creates a new instance of StatisticalEvaluator.

        :param metric: Metric to use for evaluation in this component. Supported metrics are F1 and Exact Match.
        """
        if isinstance(metric, str):
            metric = StatisticalMetric.from_str(metric)
        self._metric = metric

        self._metric_function = {
            StatisticalMetric.F1: self._f1,
            StatisticalMetric.EM: self._exact_match,
            StatisticalMetric.RECALL_SINGLE_HIT: self._recall_single_hit,
            StatisticalMetric.RECALL_MULTI_HIT: self._recall_multi_hit,
            StatisticalMetric.MRR: self._mrr,
        }[self._metric]

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, metric=self._metric.value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalEvaluator":
        data["init_parameters"]["metric"] = StatisticalMetric(data["init_parameters"]["metric"])
        return default_from_dict(cls, data)

    @component.output_types(result=float)
    def run(self, labels: List[str], predictions: List[str]) -> Dict[str, Any]:
        """
        Run the StatisticalEvaluator to compute the metric between a list of predictions and a list of labels.
        Both must be list of strings of same length.

        :param predictions: List of predictions.
        :param labels: List of labels against which the predictions are compared.
        :returns: A dictionary with the following outputs:
                    * `result` - Calculated result of the chosen metric.
        """
        return {"result": self._metric_function(labels, predictions)}

    @staticmethod
    def _f1(labels: List[str], predictions: List[str]):
        """
        Measure word overlap between predictions and labels.
        """
        if len(labels) != len(predictions):
            raise ValueError("The number of predictions and labels must be the same.")

        if len(predictions) == 0:
            # We expect callers of this function already checked if predictions and labels are equal length
            return 0.0

        scores: List[float] = []
        tokenized_predictions = [pred.split() for pred in predictions]
        tokenized_labels = [label.split() for label in labels]
        for label_tokens, prediction_tokens in zip(tokenized_labels, tokenized_predictions):
            common = collections.Counter(label_tokens) & collections.Counter(prediction_tokens)
            num_same = sum(common.values())
            if len(label_tokens) == 0 or len(prediction_tokens) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                return int(label_tokens == prediction_tokens)
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(label_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            scores.append(f1)

        return np_mean(scores)

    @staticmethod
    def _exact_match(labels: List[str], predictions: List[str]) -> float:
        """
        Measure the proportion of cases where prediction is identical to the the expected label.
        """
        if len(labels) != len(predictions):
            raise ValueError("The number of predictions and labels must be the same.")

        if len(predictions) == 0:
            # We expect callers of this function already checked if predictions and labels are equal length
            return 0.0
        score_list = np_array(predictions) == np_array(labels)
        return np_mean(score_list)

    @staticmethod
    def _recall_single_hit(labels: List[str], predictions: List[str]) -> float:
        """
        Measures how many times a label is present in at least one prediction.
        If the same label is found in multiple predictions it is only counted once.
        """
        if len(labels) == 0:
            return 0.0

        # In Recall Single Hit we only consider if a label is present in at least one prediction.
        # No need to count multiple occurrences of the same label in different predictions
        retrieved_labels = {l for l, p in itertools.product(labels, predictions) if l in p}
        return len(retrieved_labels) / len(labels)

    @staticmethod
    def _recall_multi_hit(labels: List[str], predictions: List[str]) -> float:
        """
        Measures how many times a label is present in at least one or more predictions.
        """
        if len(labels) == 0:
            return 0.0

        correct_retrievals = 0
        for label, prediction in itertools.product(labels, predictions):
            if label in prediction:
                correct_retrievals += 1

        return correct_retrievals / len(labels)

    @staticmethod
    def _mrr(labels: List[str], predictions: List[str]) -> float:
        """
        Measures the mean reciprocal rank of times a label is present in at least one or more predictions.
        """
        if len(labels) == 0:
            return 0.0

        mrr_sum = 0.0
        for label, prediction in zip(labels, predictions):
            if label in prediction:
                mrr_sum += 1 / (predictions.index(prediction) + 1)
                break

        return mrr_sum / len(labels)
