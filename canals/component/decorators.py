# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access

import logging
import inspect
from functools import wraps


logger = logging.getLogger(__name__)


def save_init_params(init_func):
    """
    Decorator that saves the init parameters of a component in `self.init_parameters`
    """

    @wraps(init_func)
    def wrapper_saveinit_parameters(self, *args, **kwargs):

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

    return wrapper_saveinit_parameters


def init_defaults(init_func):
    """
    Decorator that makes sure the `self.defaults` dictionary exists
    """

    @wraps(init_func)
    def wrapper_create_defaults_dict(self, *args, **kwargs):

        # Call the actual __init__ function with the arguments
        init_func(self, *args, **kwargs)

        # Makes sure the component has a defaults dictionary the pipeline can use
        if not hasattr(self, "defaults"):
            self.defaults = {}

    return wrapper_create_defaults_dict
