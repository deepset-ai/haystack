# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import statistics
from dataclasses import dataclass, field


def _compute_metrics(times: list[float]) -> "PipelineBenchmarkMetrics":
    """
    Computes benchmark metrics from a list of execution times (in seconds).

    :param times: A list of execution times collected across benchmark runs.
    :returns: A `PipelineBenchmarkMetrics` instance with p50, p90, p99, avg, and total.
    """
    if len(times) == 1:
        # quantiles needs at least 2 data points; handle single-run edge case
        val = times[0]
        return PipelineBenchmarkMetrics(p50=val, p90=val, p99=val, avg=val, total=val)

    quartiles = statistics.quantiles(times, n=4, method="inclusive")
    percentiles = statistics.quantiles(times, n=100, method="inclusive")

    return PipelineBenchmarkMetrics(
        p50=quartiles[1], p90=percentiles[89], p99=percentiles[98], avg=statistics.mean(times), total=sum(times)
    )


@dataclass
class PipelineBenchmarkMetrics:
    """
    Pipeline benchmark Metrics.

    :param p50: 50th percentile (median) execution time.
    :param p90: 90th percentile execution time.
    :param p99: 99th percentile execution time.
    :param avg: Arithmetic mean execution time.
    :param total: Sum of all execution times across runs.
    """

    p50: float
    p90: float
    p99: float
    avg: float
    total: float


@dataclass
class PipelineBenchmarkResult:
    """
    Aggregated benchmark results from multiple executions.

    :param pipeline: Aggregated timing metrics for the entire pipeline.
    :param components: Per-component timing metrics.
    :param slowest_component: Slowest component with the highest average execution time.
    :param fastest_run: Minimum total pipeline execution time observed (seconds).
    :param slowest_run: Maximum total pipeline execution time observed (seconds).
    :param num_runs: Number of benchmark iterations performed (excluding warmup runs).
    """

    pipeline: PipelineBenchmarkMetrics
    components: dict[str, PipelineBenchmarkMetrics] = field(default_factory=dict)
    slowest_component: str = ""
    fastest_run: float = 0.0
    slowest_run: float = 0.0
    num_runs: int = 0

    def display(self) -> None:
        """
        Display formatted summary of the benchmark results.
        """
        _MS = 1000  # seconds → milliseconds multiplier

        def _fmt(val: float) -> str:
            return f"{val * _MS:>10.3f} ms"

        header = f"{'':30s} {'p50':>14} {'p90':>14} {'p99':>14} {'avg':>14} {'total':>14}"
        separator = "-" * len(header)

        print("\n" + "=" * len(header))
        print(" Pipeline Benchmark Results")
        print("=" * len(header))

        # Pipeline level statistics
        print("\n Pipeline")
        print(separator)
        print(header)
        print(separator)
        m = self.pipeline
        print(f"  {'pipeline':<28}{_fmt(m.p50)}{_fmt(m.p90)}{_fmt(m.p99)}{_fmt(m.avg)}{_fmt(m.total)}")

        # Per component level statistics
        if self.components:
            print("\n Components  (sorted by avg, slowest first)")
            print(separator)
            print(header)
            print(separator)
            sorted_components = sorted(self.components.items(), key=lambda kv: kv[1].avg, reverse=True)
            for name, cm in sorted_components:
                label = f"  {name:<28}"
                print(f"{label}{_fmt(cm.p50)}{_fmt(cm.p90)}{_fmt(cm.p99)}{_fmt(cm.avg)}{_fmt(cm.total)}")

        # Summary statistics
        print(separator)
        print(f"\n  Runs          : {self.num_runs}")
        print(f"  Fastest run   : {self.fastest_run * _MS:.3f} ms")
        print(f"  Slowest run   : {self.slowest_run * _MS:.3f} ms")
        if self.slowest_component:
            print(f"  Slowest component : {self.slowest_component}")
        print("=" * len(header) + "\n")
