import os
import json

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
)


def parse_benchmark_files(folder_path):
    metrics = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                indexing_metrics = data.get("indexing", {})
                querying_metrics = data.get("querying")
                if indexing_metrics.get("error") is None and querying_metrics.get("error") is None:
                    metrics[filename.split(".json")[0]] = {"indexing": indexing_metrics, "querying": querying_metrics}
    return metrics


def get_model_tag(querying_metrics):
    model = querying_metrics.get("reader_model")
    model = model.lower() if model != None else model
    if model == "deepset/tinyroberta-squad2":
        return ReaderModelTags.tinyroberta

    if model == "deepset/deberta-v3-base-squad2":
        return ReaderModelTags.debertabase

    if model == "deepset/deberta-v3-large-squad2":
        return ReaderModelTags.debertalarge

    return NoneTag.none


def get_retriever_tag(name):
    name_lower = name.lower()
    if "bm25" in name_lower:
        return RetrieverModelTags.bm25

    if "minilm" in name_lower:
        return RetrieverModelTags.minilm

    if "mpnetbase" in name_lower:
        return RetrieverModelTags.mpnetbase

    return NoneTag.none


def get_documentstore_tag(querying_metrics):
    doc_store = querying_metrics.get("doc_store")
    doc_store = doc_store.lower() if doc_store != None else doc_store
    if doc_store == "opensearchdocumentstore":
        return DocumentStoreModelTags.opensearch

    if doc_store == "weaviatedocumentstore":
        return DocumentStoreModelTags.weaviate

    if doc_store == "elasticsearchdocumentstore":
        return DocumentStoreModelTags.elasticsearch

    return NoneTag.none


def get_benchmark_type_tag(model_tag, retriever_tag, document_store_tag):
    if model_tag != NoneTag.none and retriever_tag != NoneTag.none and document_store_tag != NoneTag.none:
        return BenchmarkType.retriever_reader
    elif retriever_tag != NoneTag.none and document_store_tag != NoneTag.none:
        return BenchmarkType.retriever
    elif model_tag != NoneTag.none and retriever_tag == NoneTag.none:
        return BenchmarkType.reader

    LOGGER.warn(
        f"Did not find benchmark_type for the combination of tags, retriever={retriever_tag}, reader={model_tag}, document_store={document_store_tag}"
    )
    return NoneTag.none


folder_path = "out"
benchmark_metrics = parse_benchmark_files(folder_path)
metrics_to_send_to_dd = []
for benchmark_name, metrics in benchmark_metrics.items():
    indexing_metrics = metrics["indexing"]
    querying_metrics = metrics["querying"]

    docs_per_second = indexing_metrics.get("docs_per_second")

    exact_match = querying_metrics.get("exact_match")
    f1_score = querying_metrics.get("f1")
    recall = querying_metrics.get("recall")
    seconds_per_query = querying_metrics.get("seconds_per_query")
    map_query = querying_metrics.get("map")

    size_tag = DatasetSizeTags.size_100k
    model_tag = get_model_tag(querying_metrics)
    retriever_tag = get_retriever_tag(benchmark_name)
    document_store_tag = get_documentstore_tag(querying_metrics)
    benchmark_type_tag = get_benchmark_type_tag(model_tag, retriever_tag, document_store_tag)

    tags = [size_tag, model_tag, retriever_tag, document_store_tag, benchmark_type_tag]

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

api = MetricsAPI()
api.send_custom_dd_metrics(metrics_to_send_to_dd)
