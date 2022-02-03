import json
from pathlib import Path
import os
import sys

sys.path.append("../../../../")

rest_path = Path("../../../../rest_api").absolute()
pipeline_path = str(rest_path / "pipeline" / "pipeline_empty.yaml")
app_path = str(rest_path / "application.py")
print(f"Loading OpenAPI specs from {app_path} with pipeline at {pipeline_path}")

os.environ["PIPELINE_YAML_PATH"] = pipeline_path

from rest_api.application import get_openapi_specs

# Generate the openapi specs
specs = get_openapi_specs()

# Dump the specs into a JSON file
with open(f"openapi.json", "w") as f:
    json.dump(specs, f)
