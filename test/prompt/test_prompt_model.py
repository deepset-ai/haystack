from unittest.mock import patch, Mock

import pytest

from haystack.nodes.prompt.prompt_model import PromptModel
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer, HFLocalInvocationLayer

from .conftest import create_mock_layer_that_supports


@pytest.mark.unit
def test_constructor_with_default_model():
    mock_layer = create_mock_layer_that_supports("google/flan-t5-base")
    another_layer = create_mock_layer_that_supports("another-model")

    with patch.object(PromptModelInvocationLayer, "invocation_layer_providers", new=[mock_layer, another_layer]):
        model = PromptModel()
        mock_layer.assert_called_once()
        another_layer.assert_not_called()
        model.model_invocation_layer.model_name_or_path = "google/flan-t5-base"


@pytest.mark.unit
def test_construtor_with_custom_model():
    mock_layer = create_mock_layer_that_supports("some-model")
    another_layer = create_mock_layer_that_supports("another-model")

    with patch.object(PromptModelInvocationLayer, "invocation_layer_providers", new=[mock_layer, another_layer]):
        model = PromptModel("another-model")
        mock_layer.assert_not_called()
        another_layer.assert_called_once()
        model.model_invocation_layer.model_name_or_path = "another-model"


@pytest.mark.unit
def test_constructor_with_no_supported_model():
    with pytest.raises(ValueError, match="Model some-random-model is not supported"):
        PromptModel("some-random-model")
