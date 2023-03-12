from typing import List, Optional

import os
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from haystack.v2.rest_api.app import get_app
from haystack.v2.rest_api.config import FILE_UPLOAD_PATH


logger = logging.getLogger(__name__)
router = APIRouter()
app: FastAPI = get_app()


@router.post("/files")
def upload_file(file: UploadFile = File(...), folder: Optional[str] = None):
    """
    You can use this endpoint to upload a file. It gets stored in an internal folder, ready
    to be used by Pipelines. The folder where the files are stored can be configured
    through the env var `haystack.v2.rest_api_FILE_UPLOAD_PATH` and defaults to the `files/` folder
    under the haystack.v2.rest_api installation path.

    You can reference them with the path after upload within the REST API,
    for example as `/files/<folder>/<filename>`.
    """
    upload_path = FILE_UPLOAD_PATH / folder if folder else FILE_UPLOAD_PATH

    if not os.path.exists(upload_path):
        logger.info("Creating %s", upload_path.absolute())
        os.makedirs(upload_path)

    if os.path.exists(Path(upload_path) / file.filename):
        raise HTTPException(
            status_code=409, detail=f"A file with the same name already exist. Rename it and try again."  # 409 Conflict
        )
    try:
        file_path = Path(upload_path) / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()


@router.get("/files")
def list_files(folder: Optional[str] = None):
    """
    Returns a list of the uploaded files.
    """
    if not os.path.exists(FILE_UPLOAD_PATH):
        logger.info("Creating %s", FILE_UPLOAD_PATH.absolute())
        os.makedirs(FILE_UPLOAD_PATH)

    upload_path = FILE_UPLOAD_PATH / folder if folder else FILE_UPLOAD_PATH

    if not os.path.exists(upload_path):
        raise HTTPException(status_code=404, detail=f"The path '{folder}' does not exist")
    return {
        "files": [
            filename.name
            for filename in list(Path(upload_path).iterdir())
            if filename.is_file() and filename.name != ".gitignore"
        ],
        "folders": [folder.name for folder in list(Path(upload_path).iterdir()) if folder.is_dir()],
    }


@router.post("/files/{file_name}")
def download_file(file_name: str, folder: Optional[str] = None):
    """
    You can use this endpoint to download a file.

    You can reference them with the path after upload within the REST API,
    for example as `/files/<folder>/<filename>`.
    """
    upload_path = FILE_UPLOAD_PATH / folder if folder else FILE_UPLOAD_PATH

    if not os.path.exists(upload_path):
        raise HTTPException(status_code=404, detail=f"The path '{folder}' does not exist.")

    if not os.path.exists(upload_path / file_name):
        raise HTTPException(
            status_code=404,
            detail=f"The file '{(upload_path / file_name).relative_to(FILE_UPLOAD_PATH)}' does not exist in this folder.",
        )
    return FileResponse(upload_path / file_name)
