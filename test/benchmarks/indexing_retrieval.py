from os import pipe

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.converters import TextFileToDocument
from elasticsearch_haystack import ElasticsearchDocumentStore
from elasticsearch_haystack.bm25_retriever import ElasticsearchBM25Retriever
from haystack.components.retrievers import InMemoryBM25Retriever
from utils import get_docs, get_queries
from time import perf_counter
import datetime
from haystack.document_stores import DuplicatePolicy
from typing import Dict, Any, List, Optional
import logging


paths = get_docs("msmarco.1000")

queries = get_queries("msmarco.1000")


# document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")

with open("./pipelines/elasticsearch_indexing.yaml", "r") as f:
    indexing_pipe = Pipeline.loads(f.read())

with open("./pipelines/elasticsearch_retrieval.yaml", "r") as f:
    retrieval_pipe = Pipeline.loads(f.read())


def benchmark_indexing(pipeline: Pipeline, paths: List[str]):
    start_time = perf_counter()
    pipeline.run({"converter": {"sources": paths}})
    end_time = perf_counter()

    indexing_time = end_time - start_time
    n_docs = len(paths)

    doc_store_type = pipeline.get_component("writer").document_store

    results = {
        "doc_store": doc_store_type,
        "n_docs": n_docs,
        "indexing_time": indexing_time,
        "docs_per_second": n_docs / indexing_time,
        "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": None,
    }

    return results


def benchmark_retrieval(pipeline: Pipeline, queries: List[str]):
    start_time = perf_counter()
    for query in queries:
        pipeline.run({"retriever": {"query": query}})
    end_time = perf_counter()

    querying_time = end_time - start_time

    docstore = pipeline.get_component("retriever")._document_store
    n_docs = docstore.count_documents()
    doc_store_type = type(docstore).__name__
    retriever_type = type(pipeline.get_component("retriever")).__name__

    results = {
        "retriever": retriever_type,
        "doc_store": doc_store_type,
        "n_docs": n_docs,
        "n_queries": len(queries),
        "querying_time": querying_time,
        "queries_per_second": len(queries) / querying_time,
        "seconds_per_query": querying_time / len(queries),
        # "recall": metrics["recall_single_hit"],
        # "map": metrics["map"],
        # "top_k": retriever_top_k,
        "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": None,
    }

    return results


if __name__ == "__main__":
    # print(benchmark_retrieval(retrieval_pipe, queries))
    pass
