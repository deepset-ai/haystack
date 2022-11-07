import os
import logging
import sysconfig
import socket

from pathlib import Path
from requests.exceptions import HTTPError

logger = logging.getLogger("hatch_autorun")


class GuardedSocket(socket.socket):
    """
    A socket that raises upon creation to prevent
    network access.
    """

    def __new__(cls, family=-1, type=-1, proto=-1, fileno=None):
        raise IOError()


def gen_schema(guard_socket, main_only):
    if guard_socket:
        # Prevent 3rd party libraries from calling external services while
        # we generate the schema
        socket.socket = GuardedSocket

    # import Haystack after setting up socket
    from haystack.nodes._json_schema import update_json_schema

    logger.warning(
        "Haystack is generating the YAML schema for Pipelines validation. This only happens once, after installing the package."
    )
    update_json_schema(main_only=main_only)

    # Destroy the hatch-autorun hook if it exists (needs to run just once after installation)
    try:
        os.remove(Path(sysconfig.get_paths()["purelib"]) / "hatch_autorun_farm_haystack.pth")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    try:
        gen_schema(guard_socket=True, main_only=True)
    except Exception as e:
        logger.exception("Could not generate the Haystack Pipeline schemas.", e)
