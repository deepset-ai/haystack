# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import logging
import inspect
from functools import wraps
from dataclasses import fields, dataclass

from canals.component.decorators import save_init_params


logger = logging.getLogger(__name__)


def uncooperative_save_init_params(init_func):
    """
    Decorator that saves the init parameters of a component in `self._init_parameters`
    """

    @wraps(init_func)
    def wrapper_save_init_parameters(self, *args, **kwargs):

        # Convert all args into kwargs
        sig = inspect.signature(init_func)
        arg_names = list(sig.parameters.keys())
        if any(arg_names) and arg_names[0] in ["self", "cls"]:
            arg_names.pop(0)
        args_as_kwargs = {arg_name: arg for arg, arg_name in zip(args, arg_names)}

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        _init_parameters = {**args_as_kwargs, **kwargs}
        if hasattr(self, "_init_parameters"):
            _init_parameters = {**_init_parameters, **self._init_parameters}
        self._init_parameters = _init_parameters

    return wrapper_save_init_parameters


class Optionalize(type):
    """
    Makes all the fields of the dataclass optional by setting None as their default value.
    """

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__.__func__.__defaults__ = tuple(None for _ in inspect.signature(obj.__init__).parameters)
        obj.__init__(*args, **kwargs)
        return obj


class ComponentInput(metaclass=Optionalize):
    """
    Represents the input of a component.
    """

    def names(self):
        return [field.name for field in fields(self)]

    def to_dict(self):
        return self.__dict__


class ComponentOutput:
    """
    Represents the output of a component.
    """

    def names(self):
        return [field.name for field in fields(self)]

    def to_dict(self):
        return self.__dict__
