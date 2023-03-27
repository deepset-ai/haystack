from typing import Dict, Any

import logging
import time
import json

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter
import haystack
from haystack import Pipeline

from rest_api.utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from rest_api.schema import QueryRequest, QueryResponse


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


BaseConfig.arbitrary_types_allowed = True


router = APIRouter()
app: FastAPI = get_app()
query_pipeline: Pipeline = get_pipelines().get("query_pipeline", None)
concurrency_limiter = get_pipelines().get("concurrency_limiter", None)


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
    """
    Get the running Haystack version.
    """
    return {"hs_version": haystack.__version__}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    with concurrency_limiter.run():
        result = _process_request(query_pipeline, request)
        return result


def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.params or {}
    result = pipeline.run(query=request.query, params=params, debug=request.debug)

    # Ensure answers and documents exist, even if they're empty lists
    if not "documents" in result:
        result["documents"] = []
    if not "answers" in result:
        result["answers"] = []

    logger.info(
        json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    )
    return result
