import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel

from haystack.pipeline import Pipeline
from rest_api.config import PIPELINE_YAML_PATH, FILE_UPLOAD_PATH, INDEXING_PIPELINE_NAME
from rest_api.controller.utils import as_form

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    _, pipeline_config, definitions = Pipeline._read_yaml(
        path=Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME, overwrite_with_env_variables=True
    )
    # Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
    # end up with different indices. The check below prevents creation of Indexing Pipelines with FAISSDocumentStore.   
    is_faiss_present = False
    for node in pipeline_config["nodes"]:
        if definitions[node["name"]]["type"] == "FAISSDocumentStore":
            is_faiss_present = True
            break
    if is_faiss_present:
        logger.warning("Indexing Pipeline with FAISSDocumentStore is not supported with the REST APIs.")
        INDEXING_PIPELINE = None
    else:
        INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
except KeyError:
    INDEXING_PIPELINE = None
    logger.warning("Indexing Pipeline not found in the YAML configuration. File Upload API will not be available.")


os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)  # create directory for uploading files


@as_form
class FileUploadParams(BaseModel):
    remove_numeric_tables: Optional[bool] = None
    remove_whitespace: Optional[bool] = None
    remove_empty_lines: Optional[bool] = None
    remove_header_footer: Optional[bool] = None
    valid_languages: Optional[List[str]] = None
    split_by: Optional[str] = None
    split_length: Optional[int] = None
    split_overlap: Optional[int] = None
    split_respect_sentence_boundary: Optional[bool] = None


class Response(BaseModel):
    file_id: str


@router.post("/file-upload")
def file_upload(
    files: List[UploadFile] = File(...),
    meta: Optional[str] = Form("null"),  # JSON serialized string
    params: FileUploadParams = Depends(FileUploadParams.as_form)
):
    if not INDEXING_PIPELINE:
        raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")

    file_paths: list = []
    file_metas: list = []
    meta = json.loads(meta) or {}

    for file in files:
        try:
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            meta["name"] = file.filename
            file_metas.append(meta)
        finally:
            file.file.close()

    INDEXING_PIPELINE.run(
            file_paths=file_paths,
            meta=file_metas,
            params=params.dict(),
    )
