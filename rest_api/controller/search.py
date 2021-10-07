import logging
import time
from pathlib import Path

from fastapi import APIRouter

from haystack import Pipeline
from rest_api.config import PIPELINE_YAML_PATH, QUERY_PIPELINE_NAME
from rest_api.config import LOG_LEVEL, CONCURRENT_REQUEST_PER_WORKER
from rest_api.schema import QueryRequest, QueryResponse
from rest_api.controller.utils import RequestLimiter


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()


PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)
# TODO make this generic for other pipelines with different naming
RETRIEVER = PIPELINE.get_node(name="Retriever")
DOCUMENT_STORE = RETRIEVER.document_store if RETRIEVER else None
logging.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")

concurrency_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)


@router.get("/initialized")
def initialized():
    """
    This endpoint can be used during startup to understand if the 
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    with concurrency_limiter.run():
        result = _process_request(PIPELINE, request)
        return result


def _process_request(pipeline, request) -> QueryResponse:
    start_time = time.time()

    params = request.params or {}
    params["filters"] = params.get("filters") or {}
    filters = {}
    if "filters" in params:  # put filter values into a list and remove filters with null value
        for key, values in params["filters"].items():
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            filters[key] = values
    params["filters"] = filters
    result = pipeline.run(query=request.query, params=params)
    end_time = time.time()
    logger.info({"request": request.dict(), "response": result, "time": f"{(end_time - start_time):.2f}"})

    return result
