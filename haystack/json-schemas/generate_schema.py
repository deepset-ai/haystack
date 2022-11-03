import os
import sys
import logging
import sysconfig

from pathlib import Path
from unittest import mock

logger = logging.getLogger("hatch_autorun")

# prevent Pinecone from attempting an API call when loading its config
sys.modules["pinecone.config"] = mock.MagicMock()

# finally import Haystack
from haystack.nodes._json_schema import update_json_schema

try:
    logger.warning(
        "Haystack is generating the YAML schema for Pipelines validation. This only happens once, after installing the package."
    )
    update_json_schema(main_only=True)

    # Destroy the hatch-autorun hook if it exists (needs to run just once after installation)
    try:
        os.remove(Path(sysconfig.get_paths()["purelib"]) / "hatch_autorun_farm_haystack.pth")
    except FileNotFoundError:
        pass

except Exception as e:
    logger.exception("Could not generate the Haystack Pipeline schemas.", e)
