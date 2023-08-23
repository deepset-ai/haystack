import argparse
import os
import json
from typing import Dict

from metric_handler import (
    ReaderModelTags,
    NoneTag,
    RetrieverModelTags,
    DocumentStoreModelTags,
    BenchmarkType,
    LOGGER,
    DatasetSizeTags,
    IndexingDocsPerSecond,
    QueryingExactMatchMetric,
    QueryingF1Metric,
    QueryingRecallMetric,
    QueryingSecondsPerQueryMetric,
    QueryingMapMetric,
    MetricsAPI,
    Tag,
)


def parse_benchmark_files(folder_path: str) -> Dict:
    metrics = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                indexing_metrics = data.get("indexing", {})
                querying_metrics = data.get("querying")
                config = data.get("config")
                if indexing_metrics.get("error") is None and querying_metrics.get("error") is None:
                    metrics[filename.split(".json")[0]] = {
                        "indexing": indexing_metrics,
                        "querying": querying_metrics,
                        "config": config,
                    }
    return metrics


def get_reader_tag(config: Dict) -> Tag:
    for comp in config["components"]:
        if comp["name"] == "Reader":
            model = comp["params"]["model_name_or_path"]

            if model == "deepset/tinyroberta-squad2":
                return ReaderModelTags.tinyroberta

            if model == "deepset/deberta-v3-base-squad2":
                return ReaderModelTags.debertabase

            if model == "deepset/deberta-v3-large-squad2":
                return ReaderModelTags.debertalarge

    return NoneTag.none


def get_retriever_tag(config: Dict) -> Tag:
    for comp in config["components"]:
        if comp["name"] == "Retriever":
            if comp["type"] == "BM25Retriever":
                return RetrieverModelTags.bm25

            model = comp["params"]["embedding_model"]
            if "minilm" in model.lower():
                return RetrieverModelTags.minilm

            if "mpnet-base" in model.lower():
                return RetrieverModelTags.mpnetbase

    return NoneTag.none


def get_documentstore_tag(config: Dict) -> Tag:
    for comp in config["components"]:
        if comp["name"] == "DocumentStore":
            if comp["type"] == "ElasticsearchDocumentStore":
                return DocumentStoreModelTags.elasticsearch

            if comp["type"] == "WeaviateDocumentStore":
                return DocumentStoreModelTags.weaviate

            if comp["type"] == "OpenSearchDocumentStore":
                return DocumentStoreModelTags.opensearch

    return NoneTag.none


def get_benchmark_type_tag(reader_tag, retriever_tag, document_store_tag):
    if reader_tag != NoneTag.none and retriever_tag != NoneTag.none and document_store_tag != NoneTag.none:
        return BenchmarkType.retriever_reader
    elif retriever_tag != NoneTag.none and document_store_tag != NoneTag.none:
        return BenchmarkType.retriever
    elif reader_tag != NoneTag.none and retriever_tag == NoneTag.none:
        return BenchmarkType.reader

    LOGGER.warn(
        f"Did not find benchmark_type for the combination of tags, retriever={retriever_tag}, reader={reader_tag}, "
        f"document_store={document_store_tag}"
    )
    return NoneTag.none


def collect_metrics_from_json_files(folder_path):
    benchmark_metrics = parse_benchmark_files(folder_path)
    metrics_to_send_to_dd = []
    for benchmark_name, metrics in benchmark_metrics.items():
        indexing_metrics = metrics["indexing"]
        querying_metrics = metrics["querying"]
        config = metrics["config"]

        docs_per_second = indexing_metrics.get("docs_per_second")

        exact_match = querying_metrics.get("exact_match")
        f1_score = querying_metrics.get("f1")
        recall = querying_metrics.get("recall")
        seconds_per_query = querying_metrics.get("seconds_per_query")
        map_query = querying_metrics.get("map")

        size_tag = DatasetSizeTags.size_100k
        reader_tag = get_reader_tag(config)
        retriever_tag = get_retriever_tag(config)
        document_store_tag = get_documentstore_tag(config)
        benchmark_type_tag = get_benchmark_type_tag(reader_tag, retriever_tag, document_store_tag)

        tags = [size_tag, reader_tag, retriever_tag, document_store_tag, benchmark_type_tag]

        if docs_per_second:
            metrics_to_send_to_dd.append(IndexingDocsPerSecond(docs_per_second, tags))

        if exact_match or exact_match == 0:
            metrics_to_send_to_dd.append(QueryingExactMatchMetric(exact_match, tags))

        if f1_score or f1_score == 0:
            metrics_to_send_to_dd.append(QueryingF1Metric(f1_score, tags))

        if recall or recall == 0:
            metrics_to_send_to_dd.append(QueryingRecallMetric(recall, tags))

        if seconds_per_query:
            metrics_to_send_to_dd.append(QueryingSecondsPerQueryMetric(seconds_per_query, tags))

        if map_query or map_query == 0:
            metrics_to_send_to_dd.append(QueryingMapMetric(map_query, tags))

    return metrics_to_send_to_dd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str, help="Path to the folder with benchmark results")
    parser.add_argument("datadog_api_key", type=str, help="Datadog API key")
    parser.add_argument("datadog_api_host", type=str, help="Datadog API host")
    args = parser.parse_args()

    folder_path = args.folder_path
    datadog_api_key = args.datadog_api_key
    datadog_api_host = args.datadog_api_host

    metrics_to_send_to_dd = collect_metrics_from_json_files(folder_path)
    api = MetricsAPI(datadog_api_key=datadog_api_key, datadog_host=datadog_api_host)
    api.send_custom_dd_metrics(metrics_to_send_to_dd)
