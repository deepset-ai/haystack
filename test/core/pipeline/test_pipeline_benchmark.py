# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.core.component import component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import Pipeline
from haystack.core.pipeline.benchmark import PipelineBenchmarkMetrics, PipelineBenchmarkResult, _compute_metrics


@component
class DoubleIt:
    @component.output_types(value=int)
    def run(self, value: int) -> dict:
        return {"value": value * 2}


@component
class AddOne:
    @component.output_types(value=int)
    def run(self, value: int) -> dict:
        return {"value": value + 1}


@pytest.fixture
def pipeline():
    p = Pipeline()
    p.add_component("double", DoubleIt())
    p.add_component("add_one", AddOne())
    p.connect("double.value", "add_one.value")
    return p


@pytest.fixture
def default_data():
    return {"double": {"value": 3}}


class TestComputeMetrics:
    """
    This class contains unit tests for the `_compute_metrics()` method.
    """

    def test_single_run(self):
        m = _compute_metrics([0.5])

        assert m.p50 == m.p90 == m.p99 == m.avg == 0.5
        assert m.total == 0.5

    def test_multiple_runs_monotonic(self):
        times = [0.1, 0.2, 0.3, 0.4]
        m = _compute_metrics(times)

        assert m.p50 <= m.p90 <= m.p99
        assert m.avg > 0
        assert m.total == pytest.approx(sum(times))

    def test_returns_correct_type(self):
        m = _compute_metrics([0.1, 0.2])
        assert isinstance(m, PipelineBenchmarkMetrics)

    @pytest.mark.parametrize("times, expected_avg", [([1.0, 2.0, 3.0], 2.0), ([0.1, 0.2, 0.3], 0.2)])
    def test_avg_is_mean(self, times, expected_avg):
        m = _compute_metrics(times)
        assert m.avg == pytest.approx(expected_avg)


class TestPipelineBenchmark:
    """
    This class contains tests for the Pipeline.benchmark() method
    """

    def test_return_correct_type(self, pipeline, default_data):
        result = pipeline.benchmark(default_data, num_runs=3, warmup_runs=0)
        assert isinstance(result, PipelineBenchmarkResult)

    def test_num_runs_stored(self, pipeline, default_data):
        result = pipeline.benchmark(default_data, num_runs=5, warmup_runs=0)
        assert result.num_runs == 5

    def test_pipeline_metrics_present(self, pipeline, default_data):
        result = pipeline.benchmark(default_data, num_runs=3, warmup_runs=0)

        assert isinstance(result.pipeline, PipelineBenchmarkMetrics)
        assert result.pipeline.avg > 0
        assert result.pipeline.total > 0

    def test_component_metrics_present(self, pipeline, default_data):
        result = pipeline.benchmark(default_data, num_runs=3, warmup_runs=0)

        assert "double" in result.components
        assert "add_one" in result.components

    def test_collect_times_flag_reset(self, pipeline, default_data):
        pipeline.benchmark(default_data, num_runs=3, warmup_runs=0)
        assert pipeline._collect_times is False

    def test_collect_times_flag_reset_on_exception(self):
        @component
        class Boom:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                raise RuntimeError("boom")

        p = Pipeline()
        p.add_component("boom", Boom())

        with pytest.raises(PipelineRuntimeError):
            p.benchmark({"boom": {"value": 1}}, num_runs=3, warmup_runs=0)

        assert p._collect_times is False

    def test_invalid_num_runs_raises(self, pipeline, default_data):
        with pytest.raises(ValueError, match="num_runs"):
            pipeline.benchmark(default_data, num_runs=0)

    def test_invalid_warmup_runs_raises(self, pipeline, default_data):
        with pytest.raises(ValueError, match="warmup_runs"):
            pipeline.benchmark(default_data, warmup_runs=-1)

    def test_display_without_benchmark(self, pipeline, default_data):
        result = pipeline.run(default_data)
        assert isinstance(result, dict)
        assert not hasattr(result, "display")

        with pytest.raises(AttributeError):
            result.display()  # type: ignore[attr-defined]
