# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, get_args

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class InputSocket:
    name: str
    type: type
    is_optional: bool
    sender: Optional[str] = None


@dataclass
class OutputSocket:
    name: str
    type: type


def get_socket_type_desc(type_):
    """
    Assembles a readable representation of a type. Can handle primitive types, classes, and arbitrarily nested
    structures of types from the typing module.
    """
    # get_args returns something only if this type has subtypes, in which case it needs to be printed differently.
    args = get_args(type_)

    if not args:
        if isinstance(type_, type):
            if type_.__name__.startswith("typing."):
                return type_.__name__[len("typing.") :]
            return type_.__name__

        # Literals only accept instances, not classes, so we need to account for those.
        if isinstance(type_, str):
            return f"'{type_}'"  # Quote strings
        if str(type_).startswith("typing."):
            return str(type_)[len("typing.") :]
        return str(type_)

    subtypes = ", ".join([get_socket_type_desc(subtype) for subtype in args if subtype is not type(None)])
    type_name = _get_type_name(type_, args)
    return f"{type_name}[{subtypes}]"


def _get_type_name(type_, args):
    """
    Type names differ across several Python versions. This method abstracts away the differences.
    """
    # Python < 3.10 support
    if hasattr(type_, "_name"):
        type_name = type_._name  # pylint: disable=protected-access
        # Support for Optionals and Unions in Python < 3.10
        if not type_name:
            if type(None) in args:
                type_name = "Optional"
                args = [a for a in args if a is not type(None)]
            else:
                if not any(isinstance(a, type) for a in args):
                    type_name = "Literal"
                else:
                    type_name = "Union"
        return type_name
    return type_.__name__
