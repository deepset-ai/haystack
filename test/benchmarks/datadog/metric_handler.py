from enum import Enum
from time import time
from typing import Dict, List, Optional

import datadog
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
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
        self.timestamp = int(time())
        self.name = name
        self.value = value
        self.tags = self.validate_tags(tags) if tags is not None else []

    def validate_tags(self, tags: List[Tag]) -> List[Tag]:
        valid_tags: List[Tag] = []
        for tag in tags:
            if isinstance(
                tag, (DatasetSizeTags, ReaderModelTags, RetrieverModelTags, DocumentStoreModelTags, BenchmarkType)
            ):
                valid_tags.append(tag)
            elif tag != NoneTag.none:
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
    def __init__(self, datadog_api_key: str, datadog_host: str):
        self.datadog_api_key = datadog_api_key
        self.datadog_host = datadog_host

    @retry(retry=retry_if_exception_type(ConnectionError), wait=wait_fixed(5), stop=stop_after_attempt(3), reraise=True)
    def send_custom_dd_metric(self, metric: CustomDatadogMetric) -> dict:
        datadog.initialize(api_key=self.datadog_api_key, api_host=self.datadog_host)

        tags: List[str] = list(map(lambda t: str(t.value), metric.tags))
        post_metric_response: Dict = datadog.api.Metric.send(
            metric=metric.name, points=[(metric.timestamp, metric.value)], tags=tags
        )

        if post_metric_response.get("status") != "ok":
            LOGGER.error(
                f"Could not send custom metric. Retrying. metric_name={metric.name}, metric_value={metric.value}, "
                f"status={post_metric_response.get('status')}, error={post_metric_response.get('errors')}, "
                f"{post_metric_response}"
            )
            raise ConnectionError(f"Could not send custom metric. {post_metric_response}")
        else:
            LOGGER.info(
                f"Sent custom metric. metric_name={metric.name}, metric_value={metric.value}, "
                f"status={post_metric_response.get('status')}"
            )

        return post_metric_response

    def send_custom_dd_metrics(self, metrics: List[CustomDatadogMetric]) -> List[Dict]:
        responses = []
        for metric in metrics:
            try:
                response = self.send_custom_dd_metric(metric)
                responses.append(response)
            except ConnectionError as e:
                LOGGER.error(
                    f"Could not send custom metric even after retrying. "
                    f"metric_name={metric.name}, metric_value={metric.value}"
                )
        return responses
