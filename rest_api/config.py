import os

PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", "rest_api/pipeline/pipelines.yaml")
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")

FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "./file-upload")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_PATH = os.getenv("ROOT_PATH", "/")
