from typing import Dict, Any

import logging
import time
import json

from fastapi import FastAPI, APIRouter, HTTPException

from haystack.v2.rest_api.app import get_app


logger = logging.getLogger(__name__)
router = APIRouter()
app: FastAPI = get_app()


@router.get("/pipelines")
def list_pipelines():
    """
    List the names and metadata of all available pipelines.
    """
    return {pipeline_name: pipeline.metadata for pipeline_name, pipeline in app.pipelines.items()}


@router.post("/pipelines/warmup")
def warmup_all():
    """
    Warm up all pipelines.
    """
    for pipeline_name, pipeline in app.pipelines.items():
        start_time = time.time()
        pipeline.warm_up()
        logger.info(
            json.dumps(
                {"type": "warmup", "pipeline": pipeline_name, "time": f"{(time.time() - start_time):.2f}"}, default=str
            )
        )


@router.post("/pipelines/warmup/{pipeline_name}")
def warmup(pipeline_name: str):
    """
    Warm up the specified pipeline.
    """
    if not pipeline_name in app.pipelines.keys():
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline named '{pipeline_name}' not found. "
            "Available pipelines: '{', '.join(app.pipelines.keys())}'",
        )
    pipeline = app.pipelines[pipeline_name]

    start_time = time.time()
    pipeline.warm_up()
    logger.info(
        json.dumps(
            {"type": "warmup", "pipeline": pipeline_name, "time": f"{(time.time() - start_time):.2f}"}, default=str
        )
    )


@router.post("/pipelines/run/{pipeline_name}")
def run(pipeline_name: str, data: Dict[str, Any], parameters: Dict[str, Dict[str, Any]], debug: bool = False):
    """
    Runs a pipeline. Provide the same values for `data` and `parameters` as you would in Canals or Haystack.

    If the pipeline needs files, first upload them with `POST /uploads`, then reference them with the
    path after upload within the REST API, for example as `/uploads/<filename>`.
    """
    if not pipeline_name in app.pipelines.keys():
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline named '{pipeline_name}' not found. "
            f"Available pipelines: '{', '.join(app.pipelines.keys())}'",
        )
    pipeline = app.pipelines[pipeline_name]

    start_time = time.time()
    try:
        result = pipeline.run(data=data, parameters=parameters, debug=debug)
        logger.info(
            json.dumps(
                {
                    "type": "run",
                    "pipeline": pipeline_name,
                    "data": data,
                    "parameters": parameters,
                    "debug": debug,
                    "response": result,
                    "time": f"{(time.time() - start_time):.2f}",
                },
                default=str,
            )
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline '{pipeline_name}' failed. Exception: {exc}",
        )
