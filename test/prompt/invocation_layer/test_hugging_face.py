from unittest.mock import patch, Mock

import pytest
from torch import Tensor


from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from haystack.nodes.prompt.invocation_layer.hugging_face import StopWordsCriteria


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task")
def test_supports_text2text_generation_task(mock_get_task):
    mock_get_task.return_value = "text2text-generation"
    assert HFLocalInvocationLayer.supports("supported_model")

    assert not HFLocalInvocationLayer.supports("supported_model", api_key="some_key")


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task")
def test_supports_text_generation_task(mock_get_task):
    mock_get_task.return_value = "text-generation"
    assert HFLocalInvocationLayer.supports("supported_model")

    assert not HFLocalInvocationLayer.supports("supported_model", api_key="some_key")


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task")
def test_supports_with_unsupported_task(mock_get_task):
    mock_get_task.return_value = "some-unsupported-task"
    assert not HFLocalInvocationLayer.supports("supported_model")


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.hugging_face.get_task")
def test_supports_with_unsupported_model(mock_get_task):
    mock_get_task.side_effect = RuntimeError
    assert not HFLocalInvocationLayer.supports("unsupported_model")


@pytest.mark.unit
def test_stop_words_criteria_init():
    mock_batch_encoding = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = mock_batch_encoding

    StopWordsCriteria(tokenizer=mock_tokenizer, stop_words=[], device="gpu")
    mock_tokenizer.assert_called_once_with([], add_special_tokens=False, return_tensors="pt")
    mock_batch_encoding.to.assert_called_once_with("gpu")


@pytest.mark.unit
@pytest.mark.skip("Update this test after StopWordsCriteria is fixed")
def test_stop_words_criteria_call():
    mock_batch_encoding = Mock()
    mock_batch_encoding.to.return_value = {"input_ids": Tensor([[1190]]), "attention_mask": Tensor([[1]])}
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = mock_batch_encoding

    criteria = StopWordsCriteria(tokenizer=mock_tokenizer, stop_words=[], device="gpu")
