import collections
from enum import Enum
from typing import Any, Dict, List, Optional

from numpy import array as np_array
from numpy import mean as np_mean

from haystack import default_from_dict, default_to_dict
from haystack.core.component import component

from .preprocess import _preprocess_text


@component
class StatisticalEvaluator:
    """
    StatisticalEvaluator is a component that evaluates the performance of a model based on statistical metrics.
    It's usually used in QA and Retrieval Augmented Generation (RAG) pipelines to evaluate the quality of the generated answers.

    The supported metrics are:
    - F1: Measures word overlap between predictions and labels.
    - Exact Match: Measures the proportion of cases where prediction is identical to the expected label.
    """

    class Metric(Enum):
        """
        Supported metrics
        """

        F1 = "F1"
        EM = "Exact Match"

    def __init__(
        self,
        labels: List[str],
        metric: Metric,
        regexes_to_ignore: Optional[List[str]] = None,
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
    ):
        """
        Creates a new instance of StatisticalEvaluator.

        :param labels: The list of expected answers.
        :param metric: Metric to use for evaluation in this component. Supported metrics are F1 and Exact Match.
        :param regexes_to_ignore: A list of regular expressions. If provided, it removes substrings
            matching these regular expressions from both predictions and labels before comparison. Defaults to None.
        :param ignore_case: If True, performs case-insensitive comparison. Defaults to False.
        :param ignore_punctuation: If True, removes punctuation from both predictions and labels before
            comparison. Defaults to False.
        :param ignore_numbers: If True, removes numerical digits from both predictions and labels
            before comparison. Defaults to False.
        """
        self._labels = labels
        self._metric = metric
        self._regexes_to_ignore = regexes_to_ignore
        self._ignore_case = ignore_case
        self._ignore_punctuation = ignore_punctuation
        self._ignore_numbers = ignore_numbers

        self._metric_function = {
            StatisticalEvaluator.Metric.F1: self._f1,
            StatisticalEvaluator.Metric.EM: self._exact_match,
        }[self._metric]

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            labels=self._labels,
            metric=self._metric.value,
            regexes_to_ignore=self._regexes_to_ignore,
            ignore_case=self._ignore_case,
            ignore_punctuation=self._ignore_punctuation,
            ignore_numbers=self._ignore_numbers,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatisticalEvaluator":
        data["init_parameters"]["metric"] = StatisticalEvaluator.Metric(data["init_parameters"]["metric"])
        return default_from_dict(cls, data)

    @component.output_types(result=float)
    def run(self, predictions: List[str]) -> Dict[str, Any]:
        if len(predictions) != len(self._labels):
            raise ValueError("The number of predictions and labels must be the same.")

        predictions = _preprocess_text(
            predictions, self._regexes_to_ignore, self._ignore_case, self._ignore_punctuation, self._ignore_numbers
        )
        labels = _preprocess_text(
            self._labels, self._regexes_to_ignore, self._ignore_case, self._ignore_punctuation, self._ignore_numbers
        )

        return {"result": self._metric_function(labels, predictions)}

    def _f1(self, labels: List[str], predictions: List[str]):
        """
        Measure word overlap between predictions and labels.
        """
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

    def _exact_match(self, labels: List[str], predictions: List[str]) -> float:
        """
        Measure the proportion of cases where predictiond is identical to the the expected label.
        """
        if len(predictions) == 0:
            # We expect callers of this function already checked if predictions and labels are equal length
            return 0.0
        score_list = np_array(predictions) == np_array(labels)
        return np_mean(score_list)
