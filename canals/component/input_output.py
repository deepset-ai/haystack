# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import inspect
from dataclasses import fields

logger = logging.getLogger(__name__)


class BaseIODataclass:
    def names(self):
        """
        Returns the name of all the fields of this dataclass.
        """
        return [field.name for field in fields(self)]

    def to_dict(self):
        """
        Returns a dictionary representation of this dataclass.
        """
        return self.__dict__


class Optionalize(type):
    """
    Makes all the fields of the dataclass optional by setting None as their default value.
    """

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__.__func__.__defaults__ = tuple(None for _ in inspect.signature(obj.__init__).parameters)
        obj.__init__(*args, **kwargs)
        return obj


class Variadic(type):
    """
    Adds the proper checks to a variadic input dataclass and packs the args into a list for the `__init__` call.
    """

    def __call__(cls, *args, **kwargs):
        if kwargs:
            raise ValueError(f"{cls.__name__} accepts only an unnamed list of positional parameters.")

        obj = cls.__new__(cls, *args)

        if len(inspect.signature(obj.__init__).parameters) != 1:
            raise ValueError(f"{cls.__name__} accepts only one variadic positional parameter.")

        obj.__init__(list(args))
        return obj


class ComponentInput(BaseIODataclass, metaclass=Optionalize):
    """
    Represents the input of a component.
    """

    # dataclasses are uncooperative (don't call `super()`), so we need this flag to check for inheritance
    _component_input = True


class VariadicComponentInput(BaseIODataclass, metaclass=Variadic):
    """
    Represents the input of a variadic component.
    """

    # VariadicComponentInput can't inherit from ComponentInput due to metaclasses clashes
    # dataclasses are uncooperative (don't call `super()`), so we need this flag to check for inheritance
    _component_input = True
    _variadic_component_input = True


class ComponentOutput(BaseIODataclass, metaclass=Optionalize):
    """
    Represents the output of a component.
    """

    # dataclasses are uncooperative (don't call `super()`), so we need this flag to check for inheritance
    _component_output = True
