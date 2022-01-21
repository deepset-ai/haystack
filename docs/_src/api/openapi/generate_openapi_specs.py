import json
import importlib
from pathlib import Path
import os

rest_path = (Path(__file__).parent.parent.parent.parent.parent/"rest_api").absolute()
pipeline_path = str((rest_path/"pipeline"/"pipeline_empty.yaml").absolute())
app_path = (rest_path/"application.py").absolute()
print(f"Loading OpenAPI specs from {app_path} with pipeline at {pipeline_path}")

os.environ["PIPELINE_YAML_PATH"] = pipeline_path

# Magic import from the rest_api folder
import types
import importlib.machinery
loader = importlib.machinery.SourceFileLoader('a_b', str(app_path))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

# Generate the openapi specs
specs = mod.get_openapi_specs()

# Dump the specs into a JSON file
with open(f"openapi.json", "w") as f:
    json.dump(specs, f)