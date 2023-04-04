import pytest
import torch

from haystack.errors import OpenAIError
from haystack.nodes.prompt.prompt_model import PromptModel


@pytest.mark.integration
def test_create_prompt_model():
    model = PromptModel("google/flan-t5-small")
    assert model.model_name_or_path == "google/flan-t5-small"

    model = PromptModel()
    assert model.model_name_or_path == "google/flan-t5-base"

    with pytest.raises(OpenAIError):
        # davinci selected but no API key provided
        model = PromptModel("text-davinci-003")

    model = PromptModel("text-davinci-003", api_key="no need to provide a real key")
    assert model.model_name_or_path == "text-davinci-003"

    with pytest.raises(ValueError, match="Model some-random-model is not supported"):
        PromptModel("some-random-model")

    # we can also pass model kwargs to the PromptModel
    model = PromptModel("google/flan-t5-small", model_kwargs={"model_kwargs": {"torch_dtype": torch.bfloat16}})
    assert model.model_name_or_path == "google/flan-t5-small"

    # we can also pass kwargs directly, see HF Pipeline constructor
    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": torch.bfloat16})
    assert model.model_name_or_path == "google/flan-t5-small"

    # we can't use device_map auto without accelerate library installed
    with pytest.raises(ImportError, match="requires Accelerate: `pip install accelerate`"):
        model = PromptModel("google/flan-t5-small", model_kwargs={"device_map": "auto"})
        assert model.model_name_or_path == "google/flan-t5-small"


def test_create_prompt_model_dtype():
    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": "auto"})
    assert model.model_name_or_path == "google/flan-t5-small"

    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": "torch.bfloat16"})
    assert model.model_name_or_path == "google/flan-t5-small"
