from unittest.mock import MagicMock

from haystack.core.pipeline import Pipeline
from haystack.evaluation.eval import EvaluationResult
from haystack.evaluation.metrics import Metric


class TestEvaluationResult:
    def test_init(self):
        runnable = Pipeline()
        result = EvaluationResult(runnable=runnable, inputs=[], outputs=[], expected_outputs=[])

        assert result.runnable == runnable
        assert result.inputs == []
        assert result.outputs == []
        assert result.expected_outputs == []

    def test_calculate_metrics_with_supported_metric(self):
        runnable = Pipeline()
        result = EvaluationResult(runnable=runnable, inputs=[], outputs=[], expected_outputs=[])
        result._supported_metrics[Metric.RECALL] = MagicMock()
        result.calculate_metrics(metric=Metric.RECALL)

        assert result._supported_metrics[Metric.RECALL].called_once_with()

    def test_calculate_metrics_with_non_supported_metric(self):
        runnable = Pipeline()
        result = EvaluationResult(runnable=runnable, inputs=[], outputs=[], expected_outputs=[])

        unsupported_metric = MagicMock()

        result.calculate_metrics(metric=unsupported_metric, some_argument="some_value")

        assert unsupported_metric.called_once_with(some_argument="some_value")
