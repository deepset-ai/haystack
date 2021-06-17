# coding: utf8
import shutil
import uuid
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from rest_api.config import PIPELINES_DIR
from rest_api.controller.utils import PipelineHelper, PipelineSchema

router = APIRouter()
pipeline_helper = PipelineHelper(PIPELINES_DIR)


@router.get("/pipelines", response_model=List[PipelineSchema], response_model_include={'name', 'type', 'status'})
def get_pipelines():
    return pipeline_helper.get_pipelines()


@router.get('/pipelines/{name}', response_model=List[PipelineSchema], response_model_include={'name', 'type', 'status'})
def get_pipelines_by_name(name: str):
    pipelines: list = pipeline_helper.get_pipelines(name)
    if len(pipelines) == 0:
        raise HTTPException(status_code=404, detail="Pipeline not found.")
    return pipelines


@router.post('/pipelines/{name}/activate')
def activate_pipelines(name: str):
    if pipeline_helper.activate_pipeline(name) is False:
        raise HTTPException(status_code=404, detail=f"{name} pipeline does not exist.")

    return {'status': True}


@router.post("/pipelines")
def file_upload(file: UploadFile = File(...)):
    pipelines: list = pipeline_helper.get_pipelines()
    try:
        file_path = Path(PIPELINES_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    file_content: dict = pipeline_helper.parse_yaml_file(str(file_path))
    pipeline_names: List[str] = [pipeline['name'] for pipeline in file_content['pipelines']]
    pipeline_exists = list(filter(lambda pipeline: pipeline.name in pipeline_names, pipelines))
    if len(pipeline_exists) > 0:
        os.unlink(file_path)
        raise HTTPException(status_code=409, detail="Pipeline name already exist.")

    return {'status': True}
