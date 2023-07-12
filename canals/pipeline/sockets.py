# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict

import logging
from dataclasses import dataclass, fields


logger = logging.getLogger(__name__)


@dataclass
class OutputSocket:
    name: str
    type: type


@dataclass
class InputSocket:
    name: str
    type: type
    sender: Optional[str] = None


def find_input_sockets(component) -> Dict[str, InputSocket]:
    """
    Find a component's input sockets.
    """
    return {f.name: InputSocket(name=f.name, type=f.type) for f in fields(component.__canals_input__)}


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    return {f.name: OutputSocket(name=f.name, type=f.type) for f in fields(component.__canals_output__)}
