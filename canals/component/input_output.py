# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from enum import Enum
from dataclasses import fields, is_dataclass, dataclass, asdict, MISSING

logger = logging.getLogger(__name__)


def _make_fields_optional(class_: type):
    """
    Takes a dataclass definition and modifies its __init__ so that all fields have
    a default value set.
    If a field has a default factory use it to set the default value.
    If a field has neither a default factory or value default to None.
    """
    defaults = []
    for field in fields(class_):
        default = field.default
        if field.default is MISSING and field.default_factory is MISSING:
            default = None
        elif field.default is MISSING and field.default_factory is not MISSING:
            default = field.default_factory()
        defaults.append(default)
    # mypy complains we're accessing __init__ on an instance but it's not in reality.
    # class_ is a class definition and not an instance of it, so we're good.
    # Also only I/O dataclasses are meant to be passed to this function making it a bit safer.
    class_.__init__.__defaults__ = tuple(defaults)  # type: ignore


def _make_comparable(class_: type):
    """
    Overwrites the existing __eq__ method of class_ with a custom one.
    This is meant to be used only in I/O dataclasses, it takes into account
    whether the fields are marked as comparable or not.

    This is necessary since the automatically created __eq__ method in dataclasses
    also verifies the type of the class. That causes it to fail if the I/O dataclass
    is returned by a function.

    In here we don't compare the types of self and other but only their fields.
    """

    def comparator(self, other) -> bool:
        if not is_dataclass(other):
            return False

        fields_ = [f.name for f in fields(self) if f.compare]
        other_fields = [f.name for f in fields(other) if f.compare]
        if not len(fields_) == len(other_fields):
            return False

        self_dict, other_dict = asdict(self), asdict(other)
        for field in fields_:
            if not self_dict[field] == other_dict[field]:
                return False

        return True

    setattr(class_, "__eq__", comparator)


class Connection(Enum):
    INPUT = 1
    OUTPUT = 2


def _input(input_function=None):
    """
    Decorator to mark a method that returns a dataclass defining a Component's input.

    The decorated function becomes a property.
    """

    def decorator(function):
        def wrapper(self):
            class_ = function(self)
            # If the user didn't explicitly declare the returned class
            # as dataclass we do it out of convenience
            if not is_dataclass(class_):
                class_ = dataclass(class_)

            _make_comparable(class_)
            _make_fields_optional(class_)

            return class_

        # Magic field to ease some further checks, we set it in the wrapper
        # function so we access it like this <class>.<function>.fget.__canals_connection__
        wrapper.__canals_connection__ = Connection.INPUT

        # If we don't set the documentation explicitly the user wouldn't be able to access
        # since we make wrapper a property and not the original function.
        # This is not essential but a really nice to have.
        return property(fget=wrapper, doc=function.__doc__)

    # Check if we're called as @_input or @_input()
    if input_function:
        # Called with parens
        return decorator(input_function)

    # Called without parens
    return decorator


def _output(output_function=None):
    """
    Decorator to mark a method that returns a dataclass defining a Component's output.

    The decorated function becomes a property.
    """

    def decorator(function):
        def wrapper(self):
            class_ = function(self)
            if not is_dataclass(class_):
                class_ = dataclass(class_)
            _make_comparable(class_)
            return class_

        # Magic field to ease some further checks, we set it in the wrapper
        # function so we access it like this <class>.<function>.fget.__canals_connection__
        wrapper.__canals_connection__ = Connection.OUTPUT

        # If we don't set the documentation explicitly the user wouldn't be able to access
        # since we make wrapper a property and not the original function.
        # This is not essential but a really nice to have.
        return property(fget=wrapper, doc=function.__doc__)

    # Check if we're called as @_output or @_output()
    if output_function:
        # Called with parens
        return decorator(output_function)

    # Called without parens
    return decorator
