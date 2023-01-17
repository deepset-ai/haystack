from typing import Any, Dict

import os
import logging
from pathlib import Path

from haystack.pipelines.base import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.errors import PipelineConfigError

from rest_api.controller.utils import RequestLimiter


logger = logging.getLogger(__name__)

# Each instance of FAISSDocumentStore creates an in-memory FAISS index,
# the Indexing & Query Pipelines will end up with different indices for each worker.
# The same applies for InMemoryDocumentStore.
SINGLE_PROCESS_DOC_STORES = (FAISSDocumentStore, InMemoryDocumentStore)


def _load_pipeline(pipeline_yaml_path, indexing_pipeline_name):
    # Load pipeline (if available)
    try:
        pipeline = Pipeline.load_from_yaml(Path(pipeline_yaml_path), pipeline_name=indexing_pipeline_name)
        logging.info("Loaded pipeline nodes: %s", pipeline.graph.nodes.keys())
        document_store = _get_pipeline_doc_store(pipeline)
    except PipelineConfigError as e:
        pipeline, document_store = None, None
        logger.error(
            "Error loading %s pipeline from %s. \n %s\n", indexing_pipeline_name, pipeline_yaml_path, e.message
        )
    return pipeline, document_store


def _get_pipeline_doc_store(pipeline):
    document_store = pipeline.get_document_store()
    logging.info("Loaded docstore: %s", document_store)
    if isinstance(document_store, SINGLE_PROCESS_DOC_STORES):
        logger.warning("FAISSDocumentStore or InMemoryDocumentStore should only be used with 1 worker.")
    return document_store


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported

    pipelines = {}

    # Load query pipeline & document store
    query_pipeline, document_store = _load_pipeline(config.PIPELINE_YAML_PATH, config.QUERY_PIPELINE_NAME)
    pipelines["query_pipeline"] = query_pipeline
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: %s", config.CONCURRENT_REQUEST_PER_WORKER)
    pipelines["concurrency_limiter"] = concurrency_limiter

    # Load indexing pipeline
    index_pipeline, _ = _load_pipeline(config.PIPELINE_YAML_PATH, config.INDEXING_PIPELINE_NAME)
    if not index_pipeline:
        logger.warning("Indexing Pipeline is not setup. File Upload API will not be available.")
    pipelines["indexing_pipeline"] = index_pipeline

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
