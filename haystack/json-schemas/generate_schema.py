import os
import logging
import sysconfig
from pathlib import Path

from haystack.nodes._json_schema import update_json_schema

logger = logging.getLogger(__file__)

try:
    logger.warning(f"Generating the YAML schema for Haystack Pipelines... ")
    update_json_schema(main_only=True)

    # Self-destroy after first run
    try:
        os.remove(Path(sysconfig.get_paths()["purelib"]) / "hatch_autorun_farm_haystack.pth")
    except FileNotFoundError:
        pass

except Exception as e:
    logger.exception(
        "Could not generate the Haystack Pipeline schems. Will try again next time. Uninstall Haystack to stop these attempts.",
        e,
    )
