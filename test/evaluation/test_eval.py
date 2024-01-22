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

    def test_supported_metrics_contains_all_metrics(self):
        runnable = Pipeline()
        result = EvaluationResult(runnable=runnable, inputs=[], outputs=[], expected_outputs=[])

        supported_metrics = [m.name for m in result._supported_metrics.keys()]
        all_metric_names = [m.name for m in Metric]
        assert supported_metrics == all_metric_names

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
