from typing import Dict

import json
from pathlib import Path

from canals import Pipeline


def save_pipelines(pipelines: Dict[str, Pipeline], path: Path, _writer=json.dump) -> None:
    """
    Converts a dictionary of named Pipelines into a Python dictionary that can be
    written to a JSON file.

    :param pipelines: dictionary of {name: pipeline_object}
    :param path: where to write the resulting file
    :param _writer: which function to use to write the dictionary to a file.
        Use this parameter to dump to a different format like YAML, TOML, HCL, etc.
    """
    pass


def load_pipelines(path: Path, _reader=json.load) -> Dict[str, Pipeline]:
    """
    Reads the given file and returns a dictionary of named Pipelines ready to use.

    :param path: the path of the pipelines file.
    :param _reader: which function to use to read the dictionary from file.
        Use this parameter to read from a different format like YAML, TOML, HCL, etc.
    """
    return {}
