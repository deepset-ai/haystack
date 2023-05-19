from typing import Dict, Any

import logging
import time
import json

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import haystack
from haystack import Pipeline
from haystack.nodes.prompt import PromptNode

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


@router.post("/query-streaming", response_model=StreamingResponse)
def query_streaming(request: QueryRequest):
    """
    This streaming endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    with concurrency_limiter.run():
        iterator = _get_streaming_iterator(query_pipeline, request)
        if iterator == None:
            raise HTTPException(
                status_code=501, detail="The pipeline cannot support the streaming mode. The PromptNode is not found!"
            )
        return StreamingResponse(iterator, media_type="text/event-stream")


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


def _get_streaming_iterator(pipeline, request=None):
    params = request.params or {}
    components = pipeline.components
    node_name = None
    iterator = None
    for name in components.keys():
        if isinstance(components[name], PromptNode):
            node_name = name

    if node_name != None:
        streaming_param = {"stream": True, "return_iterator": True}
        params[node_name].update(streaming_param)
        # only one streaming iterator is support for rest_api
        iterator = pipeline.run(query=request.query, params=params)["iterator"][0]

    return iterator
