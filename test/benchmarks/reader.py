from time import perf_counter
from typing import Dict
from pathlib import Path
import traceback
import datetime
import logging

from haystack import Pipeline
from haystack.nodes import BaseReader
from haystack.utils import aggregate_labels
from utils import load_eval_data, get_reader_config


def benchmark_reader(pipeline: Pipeline, labels_file: Path) -> Dict:
    try:
        labels, queries = load_eval_data(labels_file)
        eval_labels = aggregate_labels(labels)
        eval_queries = []
        eval_docs = []
        for multi_label in eval_labels:
            eval_queries.append(multi_label.query)
            eval_docs.append([multi_label.labels[0].document])

        # Run querying
        start_time = perf_counter()
        # We use run_batch instead of eval_batch because we want to get pure inference time
        predictions = pipeline.run_batch(queries=eval_queries, documents=eval_docs, labels=eval_labels, debug=True)
        end_time = perf_counter()
        querying_time = end_time - start_time

        # Evaluate predictions
        eval_result = pipeline._generate_eval_result_from_batch_preds(predictions_batches=predictions)
        metrics = eval_result.calculate_metrics()["Reader"]

        reader_type, reader_model, reader_top_k = get_reader_config(pipeline)
        results = {
            "querying": {
                "exact_match": metrics["exact_match"],
                "f1": metrics["f1"],
                "n_queries": len(eval_labels),
                "querying_time": querying_time,
                "seconds_per_query": querying_time / len(eval_labels),
                "reader": reader_type,
                "reader_model": reader_model,
                "top_k": reader_top_k,
                "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": None,
            }
        }

    except Exception:
        tb = traceback.format_exc()
        logging.error("##### The following Error was raised while running querying run:")
        logging.error(tb)
        reader_type, reader_model, reader_top_k = get_reader_config(pipeline)
        results = {
            "reader": {
                "exact_match": 0.0,
                "f1": 0.0,
                "n_queries": 0,
                "querying_time": 0.0,
                "seconds_per_query": 0.0,
                "reader": reader_type,
                "reader_model": reader_model,
                "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(tb),
            }
        }

    return results
