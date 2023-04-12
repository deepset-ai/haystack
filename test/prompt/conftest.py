from unittest.mock import Mock
import os

import pytest

from haystack.nodes.prompt import PromptModel


def create_mock_layer_that_supports(model_name, response=["fake_response"]):
    """
    Create a mock invocation layer that supports the model_name and returns response.
    """

    def mock_supports(model_name_or_path, **kwargs):
        return model_name_or_path == model_name

    return Mock(**{"model_name_or_path": model_name, "supports": mock_supports, "invoke.return_value": response})


@pytest.fixture
def prompt_model(request, haystack_azure_conf):
    if request.param == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "KEY_NOT_FOUND")
        if api_key is None or api_key == "":
            api_key = "KEY_NOT_FOUND"
        return PromptModel("text-davinci-003", api_key=api_key)
    elif request.param == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "KEY_NOT_FOUND")
        if api_key is None or api_key == "":
            api_key = "KEY_NOT_FOUND"
        return PromptModel("text-davinci-003", api_key=api_key, model_kwargs=haystack_azure_conf)
    else:
        return PromptModel("google/flan-t5-base", devices=["cpu"])


@pytest.fixture
def chatgpt_prompt_model():
    api_key = os.environ.get("OPENAI_API_KEY", "KEY_NOT_FOUND")
    if api_key is None or api_key == "":
        api_key = "KEY_NOT_FOUND"
    return PromptModel("gpt-3.5-turbo", api_key=api_key)
