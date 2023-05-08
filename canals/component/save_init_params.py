# pylint: disable=protected-access

import logging
import inspect
from functools import wraps


logger = logging.getLogger(__name__)


def set_default_component_attributes(init_func):
    """
    Decorator that prepares a few default attributes for each component:
     - saves the init parameters of a component in self._init_parameters
     - makes sure the `self.defaults` dictionary exists
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
        _init_parameters = {**args_as_kwargs, **kwargs}
        if hasattr(self, "_init_parameters"):
            _init_parameters = {**_init_parameters, **self._init_parameters}
        self._init_parameters = _init_parameters

        # Makes sure the component has a defaults dictionary the pipeline can use
        if not hasattr(self, "defaults"):
            self.defaults = {}

    return wrapper_save_init_parameters
