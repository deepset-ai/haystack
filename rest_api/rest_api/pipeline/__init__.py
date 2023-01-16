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


def _setup_indexing_pipeline(pipeline_yaml_path, indexing_pipeline_name):
    # Load indexing pipeline (if available)
    try:
        indexing_pipeline = Pipeline.load_from_yaml(Path(pipeline_yaml_path), pipeline_name=indexing_pipeline_name)
        docstore = indexing_pipeline.get_document_store()
        if isinstance(docstore, SINGLE_PROCESS_DOC_STORES):
            logger.warning("FAISSDocumentStore or InMemoryDocumentStore should only be used with 1 worker.")

    except PipelineConfigError as e:
        indexing_pipeline = None
        logger.error("%s\nFile Upload API will not be available.", e.message)

    return indexing_pipeline


def setup_pipelines() -> Dict[str, Any]:
    # Re-import the configuration variables
    from rest_api import config  # pylint: disable=reimported

    pipelines = {}

    # Load query pipeline
    query_pipeline = Pipeline.load_from_yaml(Path(config.PIPELINE_YAML_PATH), pipeline_name=config.QUERY_PIPELINE_NAME)
    logging.info("Loaded pipeline nodes: %s", query_pipeline.graph.nodes.keys())
    pipelines["query_pipeline"] = query_pipeline

    # Find document store
    document_store = query_pipeline.get_document_store()
    logging.info("Loaded docstore: %s", document_store)
    pipelines["document_store"] = document_store

    # Setup concurrency limiter
    concurrency_limiter = RequestLimiter(config.CONCURRENT_REQUEST_PER_WORKER)
    logging.info("Concurrent requests per worker: %s", config.CONCURRENT_REQUEST_PER_WORKER)
    pipelines["concurrency_limiter"] = concurrency_limiter

    pipelines["indexing_pipeline"] = _setup_indexing_pipeline(config.PIPELINE_YAML_PATH, config.INDEXING_PIPELINE_NAME)

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
