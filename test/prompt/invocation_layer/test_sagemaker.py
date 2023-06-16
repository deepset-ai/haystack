import os
from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.nodes.prompt.invocation_layer import SageMakerInvocationLayer


@pytest.mark.skipif(
        not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None),
        reason="Skipping because SageMaker not configured",
    )
@pytest.mark.integration
def test_supports():
    """
    Test that supports returns True for valid SageMakerInvocationLayer
    """
    model_name_or_path = os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT")
    assert SageMakerInvocationLayer.supports(
        model_name_or_path=model_name_or_path,
    )


@pytest.mark.skipif(
        not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None),
        reason="Skipping because SageMaker not configured",
    )
@pytest.mark.integration
def test_supports_not():
    """
    Test that supports returns False for invalid SageMakerInvocationLayer
    """
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="fake_endpoint"
    )
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="fake_endpoint", profile_name="invalid-profile"
     )


# create a fixture with mocked boto3 client and session
@pytest.fixture(scope="function")
def mock_boto3_session():
    with patch("boto3.Session") as mock_client:
        yield mock_client


@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that the default constructor sets the correct values
    """

    layer = SageMakerInvocationLayer(model_name_or_path="some_fake_model", max_length=99, aws_access_key_id="some_fake_id", aws_secret_access_key="some_fake_key", aws_session_token="some_fake_token", profile_name="some_fake_profile", region_name="fake_region")

    # assert layer. == "some_fake_key"
    assert layer.max_length == 99
    assert layer.model_name_or_path == "some_fake_model"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key = "some_fake_key",
        aws_session_token="some_fake_token",
        profile_name="some_fake_profile",
        region_name="fake_region"
    )


@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    and that model_kwargs_rejected are correctly filtered out
    """
    model_kwargs = {"temperature": 0.7, "do_sample": True, "stream": True}
    model_kwargs_rejected = {"fake_param": 0.7, "another_fake_param": 1}

    layer = SageMakerInvocationLayer(
        model_name_or_path="some_fake_model", **model_kwargs, **model_kwargs_rejected
    )
    assert "temperature" in layer.model_input_kwargs
    assert "do_sample" in layer.model_input_kwargs
    assert "fake_param" not in layer.model_input_kwargs
    assert "another_fake_param" not in layer.model_input_kwargs


@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = SageMakerInvocationLayer(model_name_or_path="google/flan-t5-xxl")
    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match("No prompt provided.")


@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stop words are correctly passed to HTTP POST request
    """
    stop_words = ["but", "not", "bye"]
    layer = SageMakerInvocationLayer(model_name_or_path="some_model", api_key="fake_key")
    with patch(
        "haystack.nodes.prompt.invocation_layer.SageMakerInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')

        layer.invoke(prompt="Tell me hello", stop_words=stop_words)

    assert mock_post.called


@pytest.mark.unit
def test_ensure_token_limit_positive_mock(mock_auto_tokenizer):
    # prompt of length 5 + max_length of 3 = 8, which is less than model_max_length of 10, so no resize
    mock_tokens = ["I", "am", "a", "tokenized", "prompt"]
    mock_prompt = "I am a tokenized prompt"

    mock_auto_tokenizer.tokenize = Mock(return_value=mock_tokens)
    mock_auto_tokenizer.convert_tokens_to_string = Mock(return_value=mock_prompt)

    layer = SageMakerInvocationLayer("some_fake_endpoint", max_length=3, model_max_length=10)
    result = layer._ensure_token_limit(mock_prompt)

    assert result == mock_prompt


@pytest.mark.unit
def test_ensure_token_limit_negative_mock(mock_auto_tokenizer):
    # prompt of length 8 + max_length of 3 = 11, which is more than model_max_length of 10, so we resize to 7
    mock_tokens = ["I", "am", "a", "tokenized", "prompt", "of", "length", "eight"]
    correct_result = "I am a tokenized prompt of length"

    mock_auto_tokenizer.tokenize = Mock(return_value=mock_tokens)
    mock_auto_tokenizer.convert_tokens_to_string = Mock(return_value=correct_result)

    layer = SageMakerInvocationLayer("some_fake_endpoint", max_length=3, model_max_length=10)
    result = layer._ensure_token_limit("I am a tokenized prompt of length eight")

    assert result == correct_result


@pytest.mark.unit
def test_empty_model_name():
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        SageMakerInvocationLayer(model_name_or_path="")

