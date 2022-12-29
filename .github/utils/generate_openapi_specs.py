#!/usr/bin/env python3

import json
from pathlib import Path
import os
import sys
import shutil

import logging

logging.basicConfig(level=logging.INFO)


sys.path.append(".")
from rest_api.utils import get_openapi_specs, get_app, get_pipelines  # pylint: disable=wrong-import-position
from haystack import __version__  # pylint: disable=wrong-import-position

REST_PATH = Path("./rest_api/rest_api").absolute()
PIPELINE_PATH = str(REST_PATH / "pipeline" / "pipeline_empty.haystack-pipeline.yml")
APP_PATH = str(REST_PATH / "application.py")
DOCS_PATH = Path("./docs") / "_src" / "api" / "openapi"

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
with open(DOCS_PATH / "openapi.json", "w") as f:
    json.dump(specs, f, indent=4)

# Remove rc versions of the specs from the folder
for specs_file in os.listdir():
    if os.path.isfile(specs_file) and "rc" in specs_file and Path(specs_file).suffix == ".json":
        os.remove(specs_file)

# Add versioned copy
shutil.copy(DOCS_PATH / "openapi.json", DOCS_PATH / f"openapi-{__version__}.json")
