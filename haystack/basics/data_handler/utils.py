import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False
