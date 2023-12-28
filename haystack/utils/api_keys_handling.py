import os
from typing import Optional


def get_api_key_from_param_or_env(component_name: str, environment_variable: str, param: Optional[str] = None) -> str:
    """
    Reads an API key from a parameter or environment variable.
    :param component_name: The name of the component/document store that requires the API key.
    :param param: The parameter that can contain the API key.
    :param environment_variable: The environment variable that can contain the API key.
    :return: The API key.
    :raises ValueError: If the API key is not found.
    """
    api_key = param or os.environ.get(environment_variable)
    # we check whether api_key is None or an empty string
    if not api_key:
        msg = (
            f"{component_name} expects an API key. "
            f"Set the {environment_variable} environment variable (recommended) or pass it explicitly."
        )
        raise ValueError(msg)
    return api_key
