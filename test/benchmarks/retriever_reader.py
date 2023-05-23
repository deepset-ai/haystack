from pathlib import Path
from time import perf_counter
import logging
import datetime
import traceback
from typing import Dict

from haystack.nodes import BaseRetriever, BaseReader
from haystack import Pipeline
from haystack.utils import aggregate_labels

from retriever import benchmark_indexing
from utils import load_eval_data


def benchmark_retriever_reader(
    indexing_pipeline: Pipeline, querying_pipeline: Pipeline, documents_directory: Path, eval_set: Path
) -> Dict:
    """
    Benchmark indexing and querying on retriever-reader pipelines on a given dataset.
    :param indexing_pipeline: Pipeline for indexing documents.
    :param querying_pipeline: Pipeline for querying documents.
    :param documents_directory: Directory containing files to index.
    :param eval_set: Path to evaluation set.
    """
    # Indexing
    indexing_results = benchmark_indexing(indexing_pipeline, documents_directory)

    # Querying
    querying_results = benchmark_querying(querying_pipeline, eval_set)

    results = {"indexing": indexing_results, "querying": querying_results}
    return results


def benchmark_querying(pipeline: Pipeline, eval_set: Path) -> Dict:
    """
    Benchmark querying. This method should only be called if indexing has already been done.
    :param pipeline: Pipeline for querying documents.
    :param eval_set: Path to evaluation set.
    """
    try:
        # Load eval data
        labels, queries = load_eval_data(eval_set)
        multi_labels = aggregate_labels(labels)

        # Run querying
        start_time = perf_counter()
        predictions = pipeline.run_batch(queries=queries, labels=multi_labels, debug=True)
        end_time = perf_counter()
        querying_time = end_time - start_time

        # Evaluate predictions
        eval_result = pipeline._generate_eval_result_from_batch_preds(predictions_batches=predictions)
        metrics = eval_result.calculate_metrics()["Reader"]

        retrievers = pipeline.get_nodes_by_class(BaseRetriever)
        retriever_type = retrievers[0].__class__.__name__ if retrievers else "No component of type BaseRetriever found"
        retriever_top_k = retrievers[0].top_k if retrievers else "No component of type BaseRetriever found"
        doc_store = pipeline.get_document_store()
        doc_store_type = doc_store.__class__.__name__ if doc_store else "No DocumentStore found"
        readers = pipeline.get_nodes_by_class(BaseReader)
        reader_type = readers[0].__class__.__name__ if readers else "No component of type BaseReader found"
        reader_top_k = readers[0].top_k if readers else "No component of type BaseReader found"
        reader_model = readers[0].model_name_or_path if readers else "No component of type BaseReader found"

        results = {
            "exact_match": metrics["exact_match"],
            "f1": metrics["f1"],
            "querying_time": querying_time,
            "seconds_per_query": querying_time / len(labels),
            "n_docs": doc_store.get_document_count(),
            "n_queries": len(labels),
            "retriever": retriever_type,
            "retriever_top_k": retriever_top_k,
            "doc_store": doc_store_type,
            "reader": reader_type,
            "reader_model": reader_model,
            "reader_top_k": reader_top_k,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": None,
        }
    except Exception:
        tb = traceback.format_exc()
        logging.error("##### The following Error was raised while running querying run:")
        logging.error(tb)
        retrievers = pipeline.get_nodes_by_class(BaseRetriever)
        retriever_type = retrievers[0].__class__.__name__ if retrievers else "No component of type BaseRetriever found"
        doc_store = pipeline.get_document_store()
        doc_store_type = doc_store.__class__.__name__ if doc_store else "No DocumentStore found"
        readers = pipeline.get_nodes_by_class(BaseReader)
        reader_type = readers[0].__class__.__name__ if readers else "No component of type BaseReader found"
        results = {
            "exact_match": 0,
            "f1": 0,
            "querying_time": 0,
            "seconds_per_query": 0,
            "n_docs": 0,
            "n_queries": 0,
            "retriever": retriever_type,
            "retriever_top_k": retrievers[0].top_k if retrievers else 0,
            "doc_store": doc_store_type,
            "reader": reader_type,
            "reader_model": readers[0].model_name_or_path if readers else "No component of type BaseReader found",
            "reader_top_k": readers[0].top_k if readers else 0,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(tb),
        }

    return results
