# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from haystack.core.pipeline import AsyncPipeline, AsyncPipelineBenchmark, BenchmarkConfig, Pipeline, PipelineBenchmark
from haystack.core.pipeline.benchmark import TimingTracer
from haystack.testing.sample_components import AddFixedValue, Double
from haystack.tracing import disable_tracing, enable_tracing, tracer


@pytest.fixture
def sample_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("add_default", AddFixedValue())
    pipeline.add_component("double", Double())

    pipeline.connect("add_two", "double")
    pipeline.connect("double", "add_default")

    return pipeline


@pytest.fixture
def sample_async_pipeline() -> AsyncPipeline:
    pipeline = AsyncPipeline()
    pipeline.add_component("add_two", AddFixedValue(add=2))
    pipeline.add_component("add_default", AddFixedValue())
    pipeline.add_component("double", Double())

    pipeline.connect("add_two", "double")
    pipeline.connect("double", "add_default")

    return pipeline


@pytest.fixture
def tracing_enabled():
    tracer = TimingTracer()
    enable_tracing(tracer)
    yield
    disable_tracing()


class TestPipelineBenchmark:
    """Test the PipelineBenchmark class."""

    def test_init_pipeline_benchmark(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2, warmup_runs=2)

        benchmark = PipelineBenchmark(sample_pipeline, input_data, config)

        assert isinstance(benchmark, PipelineBenchmark)
        assert benchmark._pipeline == sample_pipeline
        assert benchmark.input_data == input_data
        assert benchmark._config == config

    def test_pipeline_benchmark_result(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=3, warmup_runs=1)

        result = PipelineBenchmark(sample_pipeline, input_data, config).run()

        assert result.num_runs == 3
        assert result.pipeline.total > 0
        assert result.fastest_run <= result.slowest_run
        assert set(result.components.keys()) == set(sample_pipeline.graph.nodes.keys())
        assert result.slowest_component in result.components

    def test_component_metrics_structure(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=3)

        result = PipelineBenchmark(sample_pipeline, input_data, config).run()

        for metrics in result.components.values():
            assert metrics.avg >= 0
            assert metrics.total >= metrics.avg
            assert metrics.p50 <= metrics.p99

    def test_zero_benchmark_runs_config(self):
        with pytest.raises(ValueError, match="BenchmarkConfig.runs must be > 0"):
            BenchmarkConfig(runs=0)

    def test_zero_benchmark_warmup_runs_config(self):
        with pytest.raises(ValueError, match="BenchmarkConfig.warmup_runs must be >= 0"):
            BenchmarkConfig(runs=1, warmup_runs=-1)

    def test_warmup_runs_not_affect_count(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2, warmup_runs=3)

        result = PipelineBenchmark(sample_pipeline, input_data, config).run()

        assert result.num_runs == 2

    def test_pipeline_benchmark_report(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2)

        result = PipelineBenchmark(sample_pipeline, input_data, config).run()
        report = result.report()

        assert "Benchmark Results" in report
        assert "pipeline" in report
        assert "name" in report

        for name in sample_pipeline.graph.nodes.keys():
            assert name in report

    def test_pipeline_benchmark_to_json(self, sample_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2)

        result = PipelineBenchmark(sample_pipeline, input_data, config).run()

        data = dataclasses.asdict(result)

        assert "pipeline" in data
        assert "components" in data
        assert data["num_runs"] == 2

    def test_benchmark_multiple_tracer(self, sample_pipeline):
        user_tracer = TimingTracer()
        enable_tracing(user_tracer)

        original_tracer = tracer.actual_tracer
        assert tracer.actual_tracer is user_tracer

        benchmark = PipelineBenchmark(sample_pipeline, {"value": 1}, BenchmarkConfig(runs=1))
        benchmark.run()

        assert tracer.actual_tracer is original_tracer, (
            "Benchmark should restore the original tracer, but it overwrote it permanently."
        )


@pytest.mark.asyncio
class TestAsyncPipelineBenchmark:
    """Test the AsyncPipelineBenchmark class."""

    async def test_async_benchmark_result(self, sample_async_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=3)

        result = await AsyncPipelineBenchmark(sample_async_pipeline, input_data, config).run()

        assert result.num_runs == 3
        assert result.pipeline.total > 0
        assert result.fastest_run <= result.slowest_run

    async def test_async_components(self, sample_async_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2)

        result = await AsyncPipelineBenchmark(sample_async_pipeline, input_data, config).run()

        assert set(result.components.keys()) == set(sample_async_pipeline.graph.nodes.keys())

    async def test_async_report(self, sample_async_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2)

        result = await AsyncPipelineBenchmark(sample_async_pipeline, input_data, config).run()

        report = result.report()

        assert "Benchmark Results" in report

    async def test_async_to_json(self, sample_async_pipeline):
        input_data = {"value": 1}
        config = BenchmarkConfig(runs=2)

        result = await AsyncPipelineBenchmark(sample_async_pipeline, input_data, config).run()

        data = dataclasses.asdict(result)

        assert "pipeline" in data
