#!/usr/bin/env python3

import json
from pathlib import Path
import os
import sys

import logging

logging.basicConfig(level=logging.INFO)


sys.path.append(".")
from rest_api.utils import get_openapi_specs, get_app, get_pipelines  # pylint: disable=wrong-import-position
from haystack import __version__  # pylint: disable=wrong-import-position

REST_PATH = Path("./rest_api/rest_api").absolute()
PIPELINE_PATH = str(REST_PATH / "pipeline" / "pipeline_empty.haystack-pipeline.yml")
APP_PATH = str(REST_PATH / "application.py")

os.environ["PIPELINE_YAML_PATH"] = PIPELINE_PATH

logging.info("Loading OpenAPI specs from %s with pipeline at %s", APP_PATH, PIPELINE_PATH)

# To initialize the app and the pipelines
get_app()
get_pipelines()

# Generate the openapi specs
specs = get_openapi_specs()
# Add `x-readme` to disable proxy and limit sample languages on documentation (see https://docs.readme.com/main/docs/openapi-extensions)
specs.update({"x-readme": {"proxy-enabled": False, "samples-languages": ["curl", "python"]}})

# Dump the specs into a JSON file
with open("openapi.json", "w") as f:
    json.dump(specs, f, indent=4)
    f.write("\n")  # We need to add a newline, otherwise there will be a conflict with end-of-file-fixer pre-commit hook
