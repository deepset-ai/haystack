from enum import Enum
import os
from time import time
from typing import Dict, List, Optional

from datadog import api, initialize
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
import logging

LOGGER = logging.getLogger(__name__)


class Tag(Enum):
    @classmethod
    def values(cls):
        return [e.value for e in cls]


class NoneTag(Tag):
    none = "none_none_none_none-1234"  # should not match any other tag


class DatasetSizeTags(Tag):
    size_100k = "dataset_size:100k"


class ReaderModelTags(Tag):
    debertabase = "reader:debertabase"
    debertalarge = "reader:debertalarge"
    tinyroberta = "reader:tinyroberta"


class RetrieverModelTags(Tag):
    bm25 = "retriever:bm25"
    minilm = "retriever:minilm"
    mpnetbase = "retriever:mpnetbase"


class DocumentStoreModelTags(Tag):
    opensearch = "documentstore:opensearch"
    elasticsearch = "documentstore:elasticsearch"
    weaviate = "documentstore:weaviate"


class BenchmarkType(Tag):
    retriever = "benchmark_type:retriever"
    retriever_reader = "benchmark_type:retriever_reader"
    reader = "benchmark_type:reader"


class CustomDatadogMetric:
    name: str
    timestamp: float
    value: float
    tags: List[Tag]

    def __init__(self, name: str, value: float, tags: Optional[List[Tag]] = None) -> None:
        self.timestamp = time()
        self.name = name
        self.value = value
        self.tags = self.validate_tags(tags)

    def validate_tags(self, tags: Optional[List[Tag]]) -> List[Tag]:
        valid_tags = []
        if tags is not None:
            for tag in tags:
                is_dataset_size_tag = tag.value in DatasetSizeTags.values()
                is_reader_model_tag = tag.value in ReaderModelTags.values()
                is_retriever_model_tag = tag.value in RetrieverModelTags.values()
                is_document_store_tag = tag.value in DocumentStoreModelTags.values()
                is_benchmark_type_tag = tag.value in BenchmarkType.values()

                if (
                    is_dataset_size_tag
                    or is_reader_model_tag
                    or is_retriever_model_tag
                    or is_document_store_tag
                    or is_benchmark_type_tag
                ):
                    valid_tags.append(tag)
                elif tag == NoneTag.none:
                    pass
                else:
                    # Log invalid tags as errors
                    LOGGER.error(f"Tag is not a valid dataset or environment tag: tag={tag}")

        return valid_tags


class IndexingDocsPerSecond(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.indexing.docs_per_second"
        super().__init__(name=name, value=value, tags=tags)


class QueryingExactMatchMetric(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.querying.exact_match"
        super().__init__(name=name, value=value, tags=tags)


class QueryingF1Metric(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.querying.f1_score"
        super().__init__(name=name, value=value, tags=tags)


class QueryingRecallMetric(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.querying.recall"
        super().__init__(name=name, value=value, tags=tags)


class QueryingMapMetric(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.querying.map"
        super().__init__(name=name, value=value, tags=tags)


class QueryingSecondsPerQueryMetric(CustomDatadogMetric):
    def __init__(self, value: float, tags: Optional[List[Tag]] = None) -> None:
        name = "haystack.benchmarks.querying.seconds_per_query"
        super().__init__(name=name, value=value, tags=tags)


class MetricsAPI:
    @retry(
        retry=retry_if_exception(AssertionError),  # type: ignore
        wait=wait_fixed(5),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def send_custom_dd_metric(self, metric: CustomDatadogMetric) -> dict:
        options = {
            "api_key": os.getenv("DD_API_KEY"),
            "app_key": os.getenv("DD_APP_KEY"),
            "api_host": os.getenv("DD_HOST"),
        }

        initialize(**options)
        tags: List[str] = list(map(lambda t: str(t.value), metric.tags))

        post_metric_response: Dict = api.Metric.send(
            metric=metric.name, points=[metric.timestamp, metric.value], tags=tags
        )

        if post_metric_response.get("status") != "ok":
            LOGGER.error(
                f"Could not send custom metric. metric_name={metric.name}, metric_value={metric.value}, status={post_metric_response.get('status')}, error={post_metric_response.get('errors')}, {post_metric_response}"
            )
        else:
            LOGGER.info(
                f"Sent custom metric. metric_name={metric.name}, metric_value={metric.value}, status={post_metric_response.get('status')}"
            )

        assert (
            post_metric_response.get("status") == "ok"
        ), f"Could not send custom metric: metric_name={metric.name}, value={metric.value}"

        return post_metric_response

    def send_custom_dd_metrics(self, metrics: List[CustomDatadogMetric]) -> List[Dict]:
        responses = []
        for metric in metrics:
            response = self.send_custom_dd_metric(metric)
            responses.append(response)
        return responses
