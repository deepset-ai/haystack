import logging
import time
import json
from pathlib import Path

from fastapi import APIRouter

import haystack
from haystack.pipelines.base import Pipeline
from rest_api.config import PIPELINE_YAML_PATH, QUERY_PIPELINE_NAME
from rest_api.config import LOG_LEVEL, CONCURRENT_REQUEST_PER_WORKER
from rest_api.schema import QueryRequest, QueryResponse
from rest_api.controller.utils import RequestLimiter


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

from pydantic import BaseConfig

BaseConfig.arbitrary_types_allowed = True

router = APIRouter()


PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)
# TODO make this generic for other pipelines with different naming
RETRIEVER = PIPELINE.get_node(name="Retriever")
DOCUMENT_STORE = RETRIEVER.document_store if RETRIEVER else None
logging.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")

concurrency_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)
logging.info("Concurrent requests per worker: {CONCURRENT_REQUEST_PER_WORKER}")


@router.get("/initialized")
def check_status():
    """
    This endpoint can be used during startup to understand if the 
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.get("/hs_version")
def haystack_version():
    return {"hs_version": haystack.__version__}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    with concurrency_limiter.run():
        result = _process_request(PIPELINE, request)
        return result


def _process_request(pipeline, request) -> QueryResponse:
    start_time = time.time()
    
    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key, value in params.items():
        if "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    result = pipeline.run(query=request.query, params=params,debug=request.debug)
    end_time = time.time()
    logger.info(json.dumps({"request": request, "response": result, "time": f"{(end_time - start_time):.2f}"}, default=str))

    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(f"Request with deprecated filter format ('\"filters\": null'). "
                       f"Remove empty filters from params to be compliant with future versions")
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(f"Request with deprecated filter format ('{key}: null'). "
                               f"Remove null values from filters to be compliant with future versions")
                continue
            elif not isinstance(values, list):
                logger.warning(f"Request with deprecated filter format ('{key}': {values}). "
                               f"Change to '{key}':[{values}]' to be compliant with future versions")
                values = [values]

            new_filters[key] = values
    return new_filters
