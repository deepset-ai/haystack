from typing import List, Optional

import os
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from haystack.preview.rest_api.app import get_app
from haystack.preview.rest_api.config import FILE_UPLOAD_PATH


logger = logging.getLogger(__name__)
router = APIRouter()
app: FastAPI = get_app()


@router.post("/files/upload/{path:path}")
def upload_file(path: Path, file: UploadFile = File(...)):
    """
    You can use this endpoint to upload a file. It gets stored in an internal folder, ready
    to be used by Pipelines. The folder where the files are stored can be configured
    through the env var `haystack.v2.rest_api_FILE_UPLOAD_PATH` and defaults to the `files/` folder
    under the haystack.v2.rest_api installation path.

    You can reference them with the path after upload within the REST API,
    for example as `/files/<folder>/<filename>`.
    """
    path = FILE_UPLOAD_PATH / path

    if not os.path.exists(path.parent):
        logger.info("Creating %s", path.parent.absolute())
        os.makedirs(path.parent)

    if os.path.exists(Path(path)):
        raise HTTPException(
            status_code=409, detail=f"A file with the same name already exist. Rename it and try again."  # 409 Conflict
        )
    try:
        file_path = Path(path)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()


@router.get("/files/list")
@router.get("/files/list/{path:path}")
def list_files(path: Path = Path(".")):
    """
    Returns a list of the uploaded files at the given path.
    """
    if not os.path.exists(FILE_UPLOAD_PATH):
        logger.info("Creating %s", FILE_UPLOAD_PATH.absolute())
        os.makedirs(FILE_UPLOAD_PATH)

    path = FILE_UPLOAD_PATH / path

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"The path '{path.relative_to(FILE_UPLOAD_PATH)}' does not exist.")

    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"'{path.relative_to(FILE_UPLOAD_PATH)}' is not a directory.")

    return {
        "files": [
            filename.name
            for filename in list(Path(path).iterdir())
            if filename.is_file() and filename.name != ".gitignore"
        ],
        "folders": [folder.name for folder in list(Path(path).iterdir()) if folder.is_dir()],
    }


@router.get("/files/download/{path:path}")
def download_file(path: Path = Path(".")):
    """
    You can use this endpoint to download a file.

    You can reference them with the path after upload within the REST API,
    for example as `/files/<folder>/<filename>`.
    """
    path = FILE_UPLOAD_PATH / path

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"'{path.relative_to(FILE_UPLOAD_PATH)}' does not exist.")

    return FileResponse(path)
