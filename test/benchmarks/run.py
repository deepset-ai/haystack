from pathlib import Path
from typing import Dict
import argparse
import json

import posthog

from haystack import Pipeline
from haystack.pipelines.config import read_pipeline_config_from_yaml

from utils import prepare_environment, contains_reader, contains_retriever
from reader import benchmark_reader
from retriever import benchmark_retriever
from retriever_reader import benchmark_retriever_reader


# Disable telemetry reports when running benchmarks
posthog.disabled = True


def run_benchmark(pipeline_yaml: Path) -> Dict:
    """
    Run benchmarking on a given pipeline. Pipeline can be a retriever, reader, or retriever-reader pipeline.
    In case of retriever or retriever-reader pipelines, indexing is also benchmarked, so the config file must
    contain an indexing pipeline as well.

    :param pipeline_yaml: Path to pipeline YAML config. The config file should contain a benchmark_config section where
                          the following parameters are specified:
                            - documents_directory: Directory containing files to index.
                            - labels_file: Path to evaluation set.
                            - data_url (optional): URL to download the data from. Downloaded data will be stored in
                                                   the directory `data/`.
    """
    pipeline_config = read_pipeline_config_from_yaml(pipeline_yaml)
    benchmark_config = pipeline_config.pop("benchmark_config", {})

    # Prepare environment
    prepare_environment(pipeline_config, benchmark_config)
    labels_file = Path(benchmark_config["labels_file"])

    querying_pipeline = Pipeline.load_from_config(pipeline_config, pipeline_name="querying")
    pipeline_contains_reader = contains_reader(querying_pipeline)
    pipeline_contains_retriever = contains_retriever(querying_pipeline)

    # Retriever-Reader pipeline
    if pipeline_contains_retriever and pipeline_contains_reader:
        documents_dir = Path(benchmark_config["documents_directory"])
        indexing_pipeline = Pipeline.load_from_config(pipeline_config, pipeline_name="indexing")

        results = benchmark_retriever_reader(indexing_pipeline, querying_pipeline, documents_dir, labels_file)

    # Retriever pipeline
    elif pipeline_contains_retriever:
        documents_dir = Path(benchmark_config["documents_directory"])
        indexing_pipeline = Pipeline.load_from_config(pipeline_config, pipeline_name="indexing")

        results = benchmark_retriever(indexing_pipeline, querying_pipeline, documents_dir, labels_file)

    # Reader pipeline
    elif pipeline_contains_reader:
        results = benchmark_reader(querying_pipeline, labels_file)

    # Unsupported pipeline type
    else:
        raise ValueError("Pipeline must be a retriever, reader, or retriever-reader pipeline.")

    pipeline_config["benchmark_config"] = benchmark_config
    results["config"] = pipeline_config
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to pipeline YAML config.")
    parser.add_argument("--output", type=str, help="Path to output file.")
    args = parser.parse_args()

    config_file = Path(args.config)
    output_file = f"{config_file.stem}_results.json" if args.output is None else args.output
    results = run_benchmark(config_file)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
