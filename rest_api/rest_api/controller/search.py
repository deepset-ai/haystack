from typing import Dict, Any

import logging
import time
import json
import threading
import queue
import inspect

from pydantic import BaseConfig
from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
import haystack
from haystack import Pipeline
from haystack.nodes.prompt.invocation_layer import TokenStreamingHandler
from haystack.nodes import PromptNode

from rest_api.utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from rest_api.schema import QueryRequest, QueryResponse

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


BaseConfig.arbitrary_types_allowed = True


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class FastAPITokenStreamingHandler(TokenStreamingHandler):
    def __init__(self, generator: ThreadedGenerator):
        self.generator = generator

    def __call__(self, token_received, **kwargs):
        self.generator.send(token_received)
        return token_received


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


@router.post("/query-streaming", response_model=QueryResponse, response_model_exclude_none=True)
async def query_streaming(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline. If the last node in the pipeline
    is a PromptNode, the output of the last note will be a streaming text. Otherwise, the output will not be streamed.
    """
    with concurrency_limiter.run():
        result = _process_streaming_request(query_pipeline, request)
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


def _process_streaming_request(pipeline, request) -> StreamingResponse:
    params = request.params or {}
    last_node_name = list(query_pipeline.graph.nodes)[-1]
    last_node_component = query_pipeline.graph.nodes.get(last_node_name)["component"]
    run_signature_args = inspect.signature(last_node_component).parameters.keys()

    def prompt_node_invocation_thread(pipeline, g, prompt):
        try:
            if "stream_hanlder" in run_signature_args:
                if last_node_name in params:
                    params[last_node_name]["stream_handler"] = FastAPITokenStreamingHandler(g)
                else:
                    params[last_node_name] = {"stream_handler": FastAPITokenStreamingHandler(g)}
            elif isinstance(last_node_component, PromptNode):
                if last_node_name in params:
                    if "invocation_context" in params[last_node_name].keys():
                        params[last_node_name]["invocation_context"]["stream_handler"] = FastAPITokenStreamingHandler(g)
                    else:
                        params[last_node_name]["invocation_context"] = {
                            "stream_handler": FastAPITokenStreamingHandler(g)
                        }
                else:
                    params[last_node_name] = {"invocation_context": {"stream_handler": FastAPITokenStreamingHandler(g)}}
            else:
                logging.warning(
                    "The last component in the pipeline is not a PromptNode or it does not accept the parameter `stream_handler`. The output will not be streamed."
                )

            pipeline.run(query=prompt, params=params)
        finally:
            g.close()

    def token_generator(prompt: str):
        g = ThreadedGenerator()
        threading.Thread(target=prompt_node_invocation_thread, args=(pipeline, g, prompt)).start()
        return g

    return StreamingResponse(token_generator(request.query), media_type="text/event-stream")
