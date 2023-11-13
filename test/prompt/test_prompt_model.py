import asyncio
from unittest.mock import patch, MagicMock

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


@pytest.mark.asyncio
async def test_ainvoke():
    def async_return(result):
        f = asyncio.Future()
        f.set_result(result)
        return f

    mock_layer = MagicMock()  # no async-defined methods, await will fail and fall back to regular `invoke`
    mock_layer.return_value.invoke.return_value = async_return("Async Bar!")
    model = PromptModel(invocation_layer_class=mock_layer)
    assert await model.ainvoke("Foo") == "Async Bar!"


@pytest.mark.asyncio
async def test_ainvoke_falls_back_to_sync():
    mock_layer = MagicMock()  # no async-defined methods, await will fail and fall back to regular `invoke`
    mock_layer.return_value.invoke.return_value = "Bar!"
    model = PromptModel(invocation_layer_class=mock_layer)
    assert await model.ainvoke("Foo") == "Bar!"
