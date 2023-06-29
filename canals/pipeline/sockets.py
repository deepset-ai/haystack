# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, Dict, Set, get_origin, get_args

import logging
from dataclasses import dataclass, fields


logger = logging.getLogger(__name__)


@dataclass
class OutputSocket:
    name: str
    types: Set[type]


@dataclass
class InputSocket:
    name: str
    types: Set[type]
    sender: Optional[str] = None


def find_input_sockets(component) -> Dict[str, InputSocket]:
    """
    Find a component's input sockets.
    """

    input_sockets = {}
    for field in fields(component.__canals_input__):
        if get_origin(field.type) is Union:
            types = {t for t in get_args(field.type) if t is not type(None)}
        else:
            types = {field.type}

        input_sockets[field.name] = InputSocket(name=field.name, types=types)

    return input_sockets


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    output_sockets = {}
    for field in fields(component.__canals_output__):
        if get_origin(field.type) is Union:
            types = {t for t in get_args(field.type) if t is not type(None)}
        else:
            types = {field.type}

        output_sockets[field.name] = OutputSocket(name=field.name, types=types)

    return output_sockets
