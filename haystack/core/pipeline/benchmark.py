# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import statistics
import time
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from typing import Any

from haystack import logging
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.pipeline import Pipeline
from haystack.lazy_imports import LazyImport
from haystack.tracing import enable_tracing
from haystack.tracing.tracer import Span, Tracer
from haystack.tracing.tracer import tracer as global_tracer

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install tabulate'") as tabulate_import:
    from tabulate import tabulate


@dataclasses.dataclass
class TimingSpan(Span):
    operation_name: str
    tags: dict = dataclasses.field(default_factory=dict)
    duration_ms: float = 0.0

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self.tags[key] = value


class TimingTracer(Tracer):
    def __init__(self) -> None:
        self.spans: list[TimingSpan] = []

    def reset(self) -> None:
        """Reset collected spans."""
        self.spans = []

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict | None = None,
        parent_span: Span | None = None,  # noqa: ARG002
    ) -> Iterator[TimingSpan]:
        """Trace execution of a code block."""

        start = time.perf_counter()
        span = TimingSpan(operation_name=operation_name, tags=dict(tags or {}))
        try:
            yield span
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000
            self.spans.append(span)

    def current_span(self) -> Span | None:
        """Return the current active span."""
        return None

    def component_spans(self) -> list[TimingSpan]:
        """Return spans for component executions."""
        return [s for s in self.spans if s.operation_name == "haystack.component.run"]

    def pipeline_span(self) -> TimingSpan | None:
        """Return span for the pipeline execution."""
        return next(
            (s for s in self.spans if s.operation_name in ("haystack.pipeline.run", "haystack.async_pipeline.run")),
            None,
        )


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for pipeline benchmarking."""

    runs: int
    warmup_runs: int = 0

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("BenchmarkConfig.runs must be > 0")
        if self.warmup_runs < 0:
            raise ValueError("BenchmarkConfig.warmup_runs must be >= 0")


@dataclasses.dataclass
class PipelineBenchmarkMetrics:
    """Performance metrics for a pipeline or component."""

    p50: float
    p90: float
    p99: float
    avg: float
    total: float


@dataclasses.dataclass
class PipelineBenchmarkResult:
    """Result of a pipeline benchmark."""

    pipeline: PipelineBenchmarkMetrics
    components: dict[str, PipelineBenchmarkMetrics]
    slowest_component: str
    fastest_run: float
    slowest_run: float
    num_runs: int
    pipeline_name: str = "Pipeline"

    def _metrics_rows(self) -> list[dict[str, str]]:
        """Convert pipeline and component metrics into tabulate-ready rows."""

        def _fmt(m: PipelineBenchmarkMetrics, name: str) -> dict[str, str]:
            return {
                "name": name,
                "p50": f"{m.p50:.3f} ms",
                "p90": f"{m.p90:.3f} ms",
                "p99": f"{m.p99:.3f} ms",
                "avg": f"{m.avg:.3f} ms",
                "total": f"{m.total:.3f} ms",
            }

        return [_fmt(self.pipeline, "pipeline"), *(_fmt(m, n) for n, m in self.components.items())]

    def report(self) -> str:
        """Generate a human-readable report of the benchmark results using tabulate."""
        tabulate_import.check()

        rows = self._metrics_rows()
        pipeline_row, *component_rows = rows

        table = tabulate(
            [pipeline_row, {}] + component_rows, headers={k: k for k in pipeline_row}, tablefmt="simple", missingval=""
        )

        summary = "\n".join(
            [
                f"  Runs               : {self.num_runs}",
                f"  Fastest run        : {self.fastest_run:.3f} ms",
                f"  Slowest run        : {self.slowest_run:.3f} ms",
                f"  Slowest component  : {self.slowest_component}",
            ]
        )

        sep = "=" * len(table.splitlines()[0])
        return "\n".join([sep, f" {self.pipeline_name} Benchmark Results", sep, "", table, "", summary, sep])

    def to_json(self) -> str:
        """Serialize the benchmark result to JSON format."""
        return json.dumps(dataclasses.asdict(self), indent=2)


def _compute_metrics(durations: list[float]) -> PipelineBenchmarkMetrics:
    if not durations:
        return PipelineBenchmarkMetrics(p50=0.0, p90=0.0, p99=0.0, avg=0.0, total=0.0)

    if len(durations) >= 4:
        q = statistics.quantiles(durations, n=100)
        p50, p90, p99 = q[49], q[89], q[98]
    else:
        p50 = statistics.median(durations)
        p90 = p99 = max(durations)

    return PipelineBenchmarkMetrics(p50=p50, p90=p90, p99=p99, avg=statistics.mean(durations), total=sum(durations))


class Benchmark:
    """
    Abstract base class for benchmarking Haystack pipelines.

    Subclasses implement `run()` for sync or async pipelines.
    Shared logic for span collection, result building, and tracing lives here.
    """

    def __init__(self, pipeline: Pipeline | AsyncPipeline, input_data: dict[str, Any], config: BenchmarkConfig) -> None:
        """
        Initialize the benchmark.

        :param pipeline: The pipeline to benchmark.
        :param input_data: The data to use for the benchmark.
        :param config: The benchmark configuration.
        """
        self._pipeline = pipeline
        self.input_data = input_data
        self._config = config
        self._tracer = TimingTracer()

    def run(self) -> PipelineBenchmarkResult | Coroutine[Any, Any, PipelineBenchmarkResult]:
        """Run the benchmark. Subclasses return either a result or a coroutine."""
        raise NotImplementedError

    def _init_tracking(self) -> tuple[list[str], dict[str, list[float]], list[float]]:
        component_names = list(self._pipeline.graph.nodes.keys())
        component_durations: dict[str, list[float]] = {n: [] for n in component_names}
        pipeline_durations: list[float] = []

        return component_names, component_durations, pipeline_durations

    @contextmanager
    def _benchmark_tracing(self) -> Iterator[None]:
        original_tracer = global_tracer.actual_tracer
        enable_tracing(self._tracer)
        try:
            yield
        finally:
            global_tracer.actual_tracer = original_tracer

    def _collect_spans(self, component_durations: dict[str, list[float]], pipeline_durations: list[float]) -> None:
        for span in self._tracer.component_spans():
            name = span.tags.get("haystack.component.name")
            if name and name in component_durations:
                component_durations[name].append(span.duration_ms)

        ps = self._tracer.pipeline_span()
        if ps:
            pipeline_durations.append(ps.duration_ms)

    def _build_result(
        self, component_names: list[str], component_durations: dict[str, list[float]], pipeline_durations: list[float]
    ) -> PipelineBenchmarkResult:
        pipeline_metrics = _compute_metrics(pipeline_durations)
        components_metrics: dict[str, PipelineBenchmarkMetrics] = {}
        slowest_component = ""
        max_avg = 0.0

        for name in component_names:
            metrics = _compute_metrics(component_durations[name])
            components_metrics[name] = metrics
            if metrics.avg > max_avg:
                max_avg = metrics.avg
                slowest_component = name

        fastest_run = min(pipeline_durations) if pipeline_durations else 0.0
        slowest_run = max(pipeline_durations) if pipeline_durations else 0.0

        pipeline_name = "AsyncPipeline" if isinstance(self._pipeline, AsyncPipeline) else "Pipeline"

        return PipelineBenchmarkResult(
            pipeline=pipeline_metrics,
            components=components_metrics,
            slowest_component=slowest_component,
            fastest_run=fastest_run,
            slowest_run=slowest_run,
            num_runs=self._config.runs,
            pipeline_name=pipeline_name,
        )


class PipelineBenchmark(Benchmark):
    """
    Benchmark a synchronous Haystack Pipeline.

    ```python
    pipeline = Pipeline()
    input_data = {"input": 1}
    benchmark_config = BenchmarkConfig(runs=20, warmup_runs=2)

    benchmark = PipelineBenchmark(pipeline, input_data, benchmark_config)
    result = benchmark.run()

    print(result.report())
    ```
    """

    def __init__(self, pipeline: Pipeline, input_data: dict[str, Any], config: BenchmarkConfig) -> None:
        super().__init__(pipeline, input_data, config)
        self._pipeline: Pipeline

    def run(self) -> PipelineBenchmarkResult:
        """Run the sync benchmark and return the results."""
        component_names, component_durations, pipeline_durations = self._init_tracking()

        self._pipeline.warm_up()
        for _ in range(self._config.warmup_runs):
            self._pipeline.run(self.input_data)

        with self._benchmark_tracing():
            for _ in range(self._config.runs):
                self._tracer.reset()
                self._pipeline.run(self.input_data)
                self._collect_spans(component_durations, pipeline_durations)

        return self._build_result(component_names, component_durations, pipeline_durations)


class AsyncPipelineBenchmark(Benchmark):
    """
    Benchmark an asynchronous Haystack AsyncPipeline.

    ```python
    pipeline = AsyncPipeline()
    input_data = {"input": 1}
    benchmark_config = BenchmarkConfig(runs=20)

    benchmark = AsyncPipelineBenchmark(pipeline, input_data, benchmark_config)
    result = await benchmark.run()

    print(result.report())
    ```
    """

    def __init__(self, pipeline: AsyncPipeline, input_data: dict[str, Any], config: BenchmarkConfig) -> None:
        super().__init__(pipeline, input_data, config)
        self._pipeline: AsyncPipeline

    async def run(self) -> PipelineBenchmarkResult:
        """Run the async benchmark and return the results."""
        component_names, component_durations, pipeline_durations = self._init_tracking()

        self._pipeline.warm_up()
        for _ in range(self._config.warmup_runs):
            await self._pipeline.run_async(self.input_data)

        with self._benchmark_tracing():
            for _ in range(self._config.runs):
                self._tracer.reset()
                await self._pipeline.run_async(self.input_data)
                self._collect_spans(component_durations, pipeline_durations)

        return self._build_result(component_names, component_durations, pipeline_durations)
