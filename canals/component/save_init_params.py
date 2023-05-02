from typing import Iterable

import logging
import inspect
from functools import wraps

from canals.errors import ComponentError


logger = logging.getLogger(__name__)


def save_init_parameters(init_func):
    """
    Decorator that saves the init parameters of a component in a dictionary.
    """

    @wraps(init_func)
    def wrapper_save_init_parameters(self, *args, **kwargs):

        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Convert all args into kwargs
        sig = inspect.signature(init_func)
        arg_names = list(sig.parameters.keys())
        if any(arg_names) and arg_names[0] in ["self", "cls"]:
            arg_names.pop(0)
        args_as_kwargs = {arg_name: arg for arg, arg_name in zip(args, arg_names)}

        # Collect and store all the init parameters, preserving whatever the components might have already added there
        init_parameters = {**args_as_kwargs, **kwargs}
        if hasattr(self, "init_parameters"):
            init_parameters = {**init_parameters, **self.init_parameters}
        self.init_parameters = init_parameters

        # Check if the component can be saved with `save_pipelines()`
        is_serializable(self.init_parameters)

    return wrapper_save_init_parameters


def is_serializable(value):
    """
    Checks that the value given can be saved to disk with a writer like `json.dump`.
    Very conservative check.
    """
    if isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, dict):
        for val in value.values():
            is_serializable(val)
    elif isinstance(value, Iterable):
        for val in value:
            is_serializable(val)
    else:
        raise ComponentError(
            f"'{value}' is not a bool, int, float, str, None, dict or iterable. "
            "It can't be passed to the `__init__()` method of a component."
        )
