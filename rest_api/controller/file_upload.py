from typing import Optional, List, Union

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel

from haystack.pipelines.base import Pipeline
from rest_api.config import PIPELINE_YAML_PATH, FILE_UPLOAD_PATH, INDEXING_PIPELINE_NAME
from rest_api.controller.utils import as_form


logger = logging.getLogger(__name__)
router = APIRouter()

try:
    pipeline_config = Pipeline._read_pipeline_config_from_yaml(Path(PIPELINE_YAML_PATH))
    pipeline_definition = Pipeline._get_pipeline_definition(
        pipeline_config=pipeline_config, pipeline_name=INDEXING_PIPELINE_NAME
    )
    definitions = Pipeline._get_component_definitions(
        pipeline_config=pipeline_config, overwrite_with_env_variables=True
    )
    # Since each instance of FAISSDocumentStore creates an in-memory FAISS index, the Indexing & Query Pipelines would
    # end up with different indices. The same applies for InMemoryDocumentStore. The check below prevents creation of
    # Indexing Pipelines with FAISSDocumentStore or InMemoryDocumentStore.
    is_faiss_or_inmemory_present = False
    for node in pipeline_definition["nodes"]:
        if (
            definitions[node["name"]]["type"] == "FAISSDocumentStore"
            or definitions[node["name"]]["type"] == "InMemoryDocumentStore"
        ):
            is_faiss_or_inmemory_present = True
            break
    if is_faiss_or_inmemory_present:
        logger.warning(
            "Indexing Pipeline with FAISSDocumentStore or InMemoryDocumentStore is not supported with the REST APIs."
        )
        INDEXING_PIPELINE = None
    else:
        INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
except KeyError:
    INDEXING_PIPELINE = None
    logger.warning("Indexing Pipeline not found in the YAML configuration. File Upload API will not be available.")


# create directory for uploading files
os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)


@as_form
class FileConverterParams(BaseModel):
    remove_numeric_tables: Optional[bool] = None
    valid_languages: Optional[List[str]] = None


@as_form
class PreprocessorParams(BaseModel):
    clean_whitespace: Optional[bool] = None
    clean_empty_lines: Optional[bool] = None
    clean_header_footer: Optional[bool] = None
    split_by: Optional[str] = None
    split_length: Optional[int] = None
    split_overlap: Optional[int] = None
    split_respect_sentence_boundary: Optional[bool] = None


class Response(BaseModel):
    file_id: str


@router.post("/file-upload")
def upload_file(
    files: List[UploadFile] = File(...),
    # JSON serialized string
    meta: Optional[str] = Form("null"),  # type: ignore
    fileconverter_params: FileConverterParams = Depends(FileConverterParams.as_form),  # type: ignore
    preprocessor_params: PreprocessorParams = Depends(PreprocessorParams.as_form),  # type: ignore
):
    """
    You can use this endpoint to upload a file for indexing
    (see https://haystack.deepset.ai/guides/rest-api#indexing-documents-in-the-haystack-rest-api-document-store).
    """
    if not INDEXING_PIPELINE:
        raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")

    file_paths: list = []
    file_metas: list = []

    meta_form = json.loads(meta) or {}  # type: ignore
    if not isinstance(meta_form, dict):
        raise HTTPException(status_code=500, detail=f"The meta field must be a dict or None, not {type(meta_form)}")

    for file in files:
        try:
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_paths.append(file_path)
            meta_form["name"] = file.filename
            file_metas.append(meta_form)
        finally:
            file.file.close()

    INDEXING_PIPELINE.run(
        file_paths=file_paths,
        meta=file_metas,
        params={
            "TextFileConverter": fileconverter_params.dict(),
            "PDFFileConverter": fileconverter_params.dict(),
            "Preprocessor": preprocessor_params.dict(),
        },
    )
