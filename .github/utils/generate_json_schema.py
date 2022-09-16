#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


sys.path.append(".")
from haystack.nodes._json_schema import update_json_schema

update_json_schema(destination_path=Path(__file__).parent.parent.parent / "haystack" / "json-schemas")
