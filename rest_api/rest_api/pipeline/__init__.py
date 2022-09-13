from typing import Any, Dict

import os
import logging
from pathlib import Path

from haystack.pipelines.base import Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.errors import PipelineConfigError

from rest_api.controller.utils import RequestLimiter


logger = logging.getLogger(__name__)

# Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
# end up with different indices. The same applies for InMemoryDocumentStore.
UNSUPPORTED_DOC_STORES = (FAISSDocumentStore, InMemoryDocumentStore)


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

    # Load indexing pipeline (if available)
    try:
        indexing_pipeline = Pipeline.load_from_yaml(
            Path(config.PIPELINE_YAML_PATH), pipeline_name=config.INDEXING_PIPELINE_NAME
        )
        docstore = indexing_pipeline.get_document_store()
        if isinstance(docstore, UNSUPPORTED_DOC_STORES):
            indexing_pipeline = None
            raise PipelineConfigError(
                "Indexing pipelines with FAISSDocumentStore or InMemoryDocumentStore are not supported by the REST APIs."
            )

    except PipelineConfigError as e:
        indexing_pipeline = None
        logger.error("%s\nFile Upload API will not be available.", e.message)

    finally:
        pipelines["indexing_pipeline"] = indexing_pipeline

    # Create directory for uploaded files
    os.makedirs(config.FILE_UPLOAD_PATH, exist_ok=True)

    return pipelines
