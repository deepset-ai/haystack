import sys
import logging

logging.basicConfig(level=logging.INFO)


sys.path.append(".")
from haystack.nodes._json_schema import update_json_schema

update_json_schema(update_index=True)
