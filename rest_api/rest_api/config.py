import logging
import os
from pathlib import Path

from enum import Enum

class LogFormatEnum(str, Enum):
    RAW = "RAW"
    JSON = "JSON"



PIPELINE_YAML_PATH = os.getenv(
    "PIPELINE_YAML_PATH", str((Path(__file__).parent / "pipeline" / "pipelines.haystack-pipeline.yml").absolute())
)
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent / "file-upload").absolute()))

LOG_LEVEL:int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
LOGGING_LOCALS_MAX_STRING = int(os.getenv("LOGGING_LOCALS_MAX_STRING", "1000"))

LOG_FORMAT: LogFormatEnum
try: 
    LOG_FORMAT = LogFormatEnum(os.getenv("LOG_FORMAT", "JSON"))
except ValueError:
    LOG_FORMAT = LogFormatEnum.RAW


ROOT_PATH = os.getenv("ROOT_PATH", "/")

CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", "4"))
