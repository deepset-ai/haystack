# coding: utf8
import os
from pathlib import Path

PIPELINES_DIR = os.getenv("PIPELINES_DIR", f"{Path(__file__).absolute().parent}/pipelines")
PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", f"{PIPELINES_DIR}/pipelines.yaml")  # need to  changed
ACTIVE_PIPELINE_FILE = f"{PIPELINES_DIR}/active.yaml"
DEFAULT_PIPELINE = os.getenv("QUERY_PIPELINE_NAME", "query")

INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "./file-upload")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
