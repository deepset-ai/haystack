# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, Dict, Set, Any, get_origin, get_args

import logging
from dataclasses import dataclass, fields, Field


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
    return {f.name: InputSocket(name=f.name, types=_get_types(f)) for f in fields(component.__canals_input__)}


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    return {f.name: OutputSocket(name=f.name, types=_get_types(f)) for f in fields(component.__canals_output__)}


def _get_types(field: Field) -> Set[Any]:
    if get_origin(field.type) is Union:
        return {t for t in get_args(field.type) if t is not type(None)}
    return {field.type}
