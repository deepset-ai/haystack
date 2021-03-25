import os

PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", "rest_api/pipelines.yaml")
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "./file-upload")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
