# coding: utf8
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from typing import List

from rest_api.config import PIPELINES_DIR
from rest_api.controller.utils import PipelineHelper, PipelineSchema

router = APIRouter()
pipeline_helper = PipelineHelper(PIPELINES_DIR)


@router.get("/pipelines", response_model=List[PipelineSchema], response_model_include={'name', 'type', 'status'})
def get_pipelines():
    return pipeline_helper.get_pipelines()


@router.get('/pipelines/{name}', response_model=List[PipelineSchema], response_model_exclude={'yaml_file'})
def get_pipelines_by_name(name: str):
    return pipeline_helper.get_pipelines(name)


@router.post('/pipelines/{name}/activate')
def activate_pipelines(name: str):
    return pipeline_helper.activate_pipeline(name)


@router.post("/pipelines")
def file_upload(file: UploadFile = File(...)):
    try:
        file_path = Path(PIPELINES_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()