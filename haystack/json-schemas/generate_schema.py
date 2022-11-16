import os
import logging
import sysconfig
import socket

from pathlib import Path

logger = logging.getLogger("hatch_autorun")


class GuardedSocket(socket.socket):
    """
    A socket that raises upon creation to prevent
    network access.
    """

    def __new__(cls, family=-1, type=-1, proto=-1, fileno=None):
        raise IOError()


original_socket = socket.socket
socket.socket = GuardedSocket

# import Haystack after setting up socket
from haystack.nodes._json_schema import update_json_schema


update_json_schema(main_only=True)
socket.socket = original_socket

# Destroy the hatch-autorun hook if it exists (needs to run just once after installation)
try:
    os.remove(Path(sysconfig.get_paths()["purelib"]) / "hatch_autorun_farm_haystack.pth")
except FileNotFoundError:
    pass

logger.warning(
    "Haystack generated the YAML schema for Pipelines validation. This only happens once, after installing the package."
)
