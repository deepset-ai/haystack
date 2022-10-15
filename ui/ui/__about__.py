import logging

from pathlib import Path


VERSION = "0.0.0"
try:
    # After git clone, VERSION.txt is in the root folder
    VERSION = open(Path(__file__).parent.parent / "VERSION.txt", "r").read()
except Exception:
    try:
        # In Docker, VERSION.txt is in the same folder
        VERSION = open(Path(__file__).parent / "VERSION.txt", "r").read()
    except Exception as e:
        logging.exception("No VERSION.txt found!")
