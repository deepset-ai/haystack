# coding: utf8
import os
from pathlib import Path

PIPELINES_DIR = os.getenv("PIPELINES_DIR", f"{Path(__file__).absolute().parent}/pipelines")
ACTIVE_PIPELINE_FILE = os.getenv("ACTIVE_PIPELINE_FILE", f"{PIPELINES_DIR}/active.yaml")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "./file-upload")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
