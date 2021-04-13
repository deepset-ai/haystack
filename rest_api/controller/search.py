import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel

from haystack import Pipeline
from rest_api.config import PIPELINE_YAML_PATH, LOG_LEVEL, QUERY_PIPELINE_NAME
from rest_api.controller.utils import RequestLimiter

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

router = APIRouter()


class Request(BaseModel):
    query: str
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None


class Answer(BaseModel):
    answer: Optional[str]
    question: Optional[str]
    score: Optional[float] = None
    probability: Optional[float] = None
    context: Optional[str]
    offset_start: int
    offset_end: int
    offset_start_in_doc: Optional[int]
    offset_end_in_doc: Optional[int]
    document_id: Optional[str] = None
    meta: Optional[Dict[str, str]]


class Response(BaseModel):
    query: str
    answers: List[Answer]


PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)
logger.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")
concurrency_limiter = RequestLimiter(4)


@router.post("/query", response_model=Response)
def query(request: Request):
    with concurrency_limiter.run():
        result = _process_request(PIPELINE, request)
        return result


def _process_request(pipeline, request) -> Response:
    start_time = time.time()

    filters = {}
    if request.filters:
        # put filter values into a list and remove filters with null value
        for key, values in request.filters.items():
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            filters[key] = values

    result = pipeline.run(query=request.query, filters=filters)

    end_time = time.time()
    logger.info(json.dumps({"request": request.dict(), "response": result, "time": f"{(end_time - start_time):.2f}"}))

    return result
