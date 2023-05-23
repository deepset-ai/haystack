# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, Dict, get_args

import logging
import inspect
from dataclasses import dataclass

from canals.component.input_output import fields


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
    run_signature = inspect.signature(component.__class__.run)

    input_annotation = run_signature.parameters["data"].annotation
    if not input_annotation or input_annotation == inspect.Parameter.empty:
        input_annotation = component.input_type
    variadic = hasattr(input_annotation, "_variadic_component_input")

    input_sockets = {}
    for field in fields(input_annotation):
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
        if variadic:
            type_ = get_args(type_)[0]

        input_sockets[field.name] = InputSocket(name=field.name, type=type_, variadic=variadic)

    return input_sockets


def find_output_sockets(component) -> Dict[str, OutputSocket]:
    """
    Find a component's output sockets.
    """
    run_signature = inspect.signature(component.run)

    return_annotation = run_signature.return_annotation
    if return_annotation == inspect.Parameter.empty:
        return_annotation = component.output_type

    output_sockets = {}
    for field in fields(return_annotation):
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
