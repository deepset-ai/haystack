import json
import importlib
from pathlib import Path
import os

rest_path = Path(__file__).parent.parent.parent.parent.parent/"rest_api"
os.environ["PIPELINE_YAML_PATH"] = str((rest_path/"pipeline"/"pipeline_empty.yaml").absolute())

# Magic import from the rest_api folder
import types
import importlib.machinery
path = rest_path/"application.py"
loader = importlib.machinery.SourceFileLoader('a_b', str(path))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

# Generate the openapi specs
specs = mod.get_openapi_specs()

# Dump the specs into a JSON file
with open(f"openapi.json", "w") as f:
    json.dump(specs, f)