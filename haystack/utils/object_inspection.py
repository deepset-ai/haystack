import inspect
from typing import Callable


def get_parameter_info(method: Callable):
    """
    Get information about the parameters of a method. Specifically, the default value and whether the parameter is
    optional or not.

    :param method: The method to get parameter information for.
    """
    signature = inspect.signature(method)
    parameter_info = {}

    for parameter in signature.parameters.values():
        parameter_name = parameter.name
        default_value = parameter.default

        if default_value is inspect.Parameter.empty:
            parameter_info[parameter_name] = {"default_value": None, "optional": False}
        else:
            parameter_info[parameter_name] = {"default_value": default_value, "optional": True}

    return parameter_info
