import json
from pathlib import Path
import os
import sys
import shutil

sys.path.append("../../../../")

rest_path = Path("../../../../rest_api").absolute()
pipeline_path = str(rest_path / "pipeline" / "pipeline_empty.yaml")
app_path = str(rest_path / "application.py")
print(f"Loading OpenAPI specs from {app_path} with pipeline at {pipeline_path}")

os.environ["PIPELINE_YAML_PATH"] = pipeline_path

from rest_api.application import get_openapi_specs, haystack_version

# Generate the openapi specs
specs = get_openapi_specs()

# Dump the specs into a JSON file
with open("openapi.json", "w") as f:
    json.dump(specs, f, indent=4)

# Remove rc versions of the specs from the folder
for specs_file in os.listdir():
    if os.path.isfile(specs_file) and "rc" in specs_file and Path(specs_file).suffix == ".json":
        os.remove(specs_file)

# Add versioned copy
shutil.copy("openapi.json", f"openapi-{haystack_version}.json")
