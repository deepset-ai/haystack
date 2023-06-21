# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, Dict, get_args

import logging
from dataclasses import dataclass, fields

from canals.component.input_output import Connection


logger = logging.getLogger(__name__)


@dataclass
class OutputSocket:
    name: str
    type: type


@dataclass
class InputSocket:
    name: str
    type: type
    variadic: bool
    taken_by: Optional[str] = None


def find_input_sockets(component) -> Dict[str, InputSocket]:
    """
    Find a component's input sockets.
    """
    is_variadic = type(component).__canals_input__.fget.__canals_connection__ is Connection.INPUT_VARIADIC

    input_sockets = {}
    for field in fields(component.__canals_input__):
        type_ = field.type
        #   Note: we're forced to use type() == type() due to an explicit limitation of the typing library,
        #   that disables `issubclass` on typing classes.
        if hasattr(type_, "__origin__") and type_.__origin__ == Union:
            if len(get_args(type_)) == 2 and type(None) in get_args(type_):
                # we support optional types, but unwrap them
                type_ = get_args(field.type)[0]
            else:
                raise ValueError("Components do not support Union types for connections yet.")
        # Unwrap List types to get the internal type if the argument is variadic
        if is_variadic:
            type_ = get_args(type_)[0]

        input_sockets[field.name] = InputSocket(name=field.name, type=type_, variadic=is_variadic)

    return input_sockets


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    output_sockets = {}
    for field in fields(component.__canals_output__):
        type_ = field.type
        #   Note: we're forced to use type() == type() due to an explicit limitation of the typing library,
        #   that disables `issubclass` on typing classes.
        if hasattr(type_, "__origin__") and type_.__origin__ == Union:
            if len(get_args(type_)) == 2 and type(None) in get_args(type_):
                # we support optional types, but unwrap them
                type_ = get_args(field.type)[0]
            else:
                raise ValueError("Components do not support Union types for connections yet.")

        output_sockets[field.name] = OutputSocket(name=field.name, type=type_)

    return output_sockets
