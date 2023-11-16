import os
from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.lazy_imports import LazyImport

from haystack.errors import SageMakerConfigurationError
from haystack.nodes.prompt.invocation_layer import SageMakerHFTextGenerationInvocationLayer

with LazyImport() as boto3_import:
    from botocore.exceptions import BotoCoreError


# create a fixture with mocked boto3 client and session
@pytest.fixture
def mock_boto3_session():
    with patch("boto3.Session") as mock_client:
        yield mock_client


@pytest.fixture
def mock_prompt_handler():
    with patch("haystack.nodes.prompt.invocation_layer.handlers.DefaultPromptHandler") as mock_prompt_handler:
        yield mock_prompt_handler


@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that the default constructor sets the correct values
    """

    layer = SageMakerHFTextGenerationInvocationLayer(
        model_name_or_path="some_fake_model",
        max_length=99,
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        aws_profile_name="some_fake_profile",
        aws_region_name="fake_region",
    )

    assert layer.max_length == 99
    assert layer.model_name_or_path == "some_fake_model"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        profile_name="some_fake_profile",
        region_name="fake_region",
    )


@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    and that model_kwargs_rejected are correctly filtered out
    """
    model_kwargs = {"temperature": 0.7, "do_sample": True, "stream": True}
    model_kwargs_rejected = {"fake_param": 0.7, "another_fake_param": 1}

    layer = SageMakerHFTextGenerationInvocationLayer(
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
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="some_fake_model")
    with pytest.raises(ValueError, match="No prompt provided."):
        layer.invoke()


@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stop words are correctly passed to HTTP POST request
    """
    stop_words = ["but", "not", "bye"]
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="some_model", api_key="fake_key")
    with patch("haystack.nodes.prompt.invocation_layer.SageMakerHFTextGenerationInvocationLayer._post") as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')

        layer.invoke(prompt="Tell me hello", stop_words=stop_words)

    assert mock_post.called
    _, call_kwargs = mock_post.call_args
    assert call_kwargs["params"]["stop"] == stop_words


@pytest.mark.unit
def test_short_prompt_is_not_truncated(mock_boto3_session):
    # Define a short mock prompt and its tokenized version
    mock_prompt_text = "I am a tokenized prompt"
    mock_prompt_tokens = mock_prompt_text.split()

    # Mock the tokenizer so it returns our predefined tokens
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = mock_prompt_tokens

    # We set a small max_length for generated text (3 tokens) and a total model_max_length of 10 tokens
    # Since our mock prompt is 5 tokens long, it doesn't exceed the
    # total limit (5 prompt tokens + 3 generated tokens < 10 tokens)
    max_length_generated_text = 3
    total_model_max_length = 10

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        layer = SageMakerHFTextGenerationInvocationLayer(
            "some_fake_endpoint", max_length=max_length_generated_text, model_max_length=total_model_max_length
        )
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)

    # The prompt doesn't exceed the limit, _ensure_token_limit doesn't truncate it
    assert prompt_after_resize == mock_prompt_text


@pytest.mark.unit
def test_long_prompt_is_truncated(mock_boto3_session):
    # Define a long mock prompt and its tokenized version
    long_prompt_text = "I am a tokenized prompt of length eight"
    long_prompt_tokens = long_prompt_text.split()

    # _ensure_token_limit will truncate the prompt to make it fit into the model's max token limit
    truncated_prompt_text = "I am a tokenized prompt of length"

    # Mock the tokenizer to return our predefined tokens
    # convert tokens to our predefined truncated text
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenize.return_value = long_prompt_tokens
    mock_tokenizer.convert_tokens_to_string.return_value = truncated_prompt_text

    # We set a small max_length for generated text (3 tokens) and a total model_max_length of 10 tokens
    # Our mock prompt is 8 tokens long, so it exceeds the total limit (8 prompt tokens + 3 generated tokens > 10 tokens)
    max_length_generated_text = 3
    total_model_max_length = 10

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        layer = SageMakerHFTextGenerationInvocationLayer(
            "some_fake_endpoint", max_length=max_length_generated_text, model_max_length=total_model_max_length
        )
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)

    # The prompt exceeds the limit, _ensure_token_limit truncates it
    assert prompt_after_resize == truncated_prompt_text


@pytest.mark.unit
def test_empty_model_name():
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        SageMakerHFTextGenerationInvocationLayer(model_name_or_path="")


@pytest.mark.unit
def test_streaming_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream parameter passed as init kwarg is correctly logged as not supported
    """
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="irrelevant", stream=True)

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(prompt="Tell me hello")


@pytest.mark.unit
def test_streaming_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream parameter passed as invoke kwarg is correctly logged as not supported
    """
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="irrelevant")

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(prompt="Tell me hello", stream=True)


@pytest.mark.unit
def test_streaming_handler_init_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream_handler parameter passed as init kwarg is correctly logged as not supported
    """
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="irrelevant", stream_handler=Mock())

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(prompt="Tell me hello")


@pytest.mark.unit
def test_streaming_handler_invoke_kwarg(mock_auto_tokenizer, mock_boto3_session):
    """
    Test stream_handler parameter passed as invoke kwarg is correctly logged as not supported
    """
    layer = SageMakerHFTextGenerationInvocationLayer(model_name_or_path="irrelevant")

    with pytest.raises(SageMakerConfigurationError, match="SageMaker model response streaming is not supported yet"):
        layer.invoke(prompt="Tell me hello", stream_handler=Mock())


@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    """
    Test that the SageMakerHFTextGenerationInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method
    """
    mock_client = MagicMock()
    mock_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    # Patch the class method to return the mock session
    with patch(
        "haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session",
        return_value=mock_session,
    ):
        supported = SageMakerHFTextGenerationInvocationLayer.supports(
            model_name_or_path="some_sagemaker_deployed_model", aws_profile_name="some_real_profile"
        )
    args, kwargs = mock_client.describe_endpoint.call_args
    assert kwargs["EndpointName"] == "some_sagemaker_deployed_model"

    args, kwargs = mock_session.client.call_args
    assert args[0] == "sagemaker-runtime"
    assert supported


@pytest.mark.unit
def test_supports_not_on_invalid_aws_profile_name():
    """
    Test that the SageMakerHFTextGenerationInvocationLayer raises SageMakerConfigurationError when the profile name is invalid
    """

    with patch("boto3.Session") as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(SageMakerConfigurationError, match="Failed to initialize the session"):
            SageMakerHFTextGenerationInvocationLayer.supports(
                model_name_or_path="some_fake_model", aws_profile_name="some_fake_profile"
            )


@pytest.mark.skipif(
    not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None), reason="Skipping because SageMaker not configured"
)
@pytest.mark.integration
def test_supports_triggered_for_valid_sagemaker_endpoint():
    """
    Test that the SageMakerHFTextGenerationInvocationLayer identifies a valid SageMaker Inference endpoint via the supports() method
    """
    model_name_or_path = os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT")
    assert SageMakerHFTextGenerationInvocationLayer.supports(model_name_or_path=model_name_or_path)


@pytest.mark.skipif(
    not os.environ.get("TEST_SAGEMAKER_MODEL_ENDPOINT", None), reason="Skipping because SageMaker not configured"
)
@pytest.mark.integration
def test_supports_not_triggered_for_invalid_iam_profile():
    """
    Test that the SageMakerHFTextGenerationInvocationLayer identifies an invalid SageMaker Inference endpoint
    (in this case because of an invalid IAM AWS Profile via the supports() method)
    """
    assert not SageMakerHFTextGenerationInvocationLayer.supports(model_name_or_path="fake_endpoint")
    assert not SageMakerHFTextGenerationInvocationLayer.supports(
        model_name_or_path="fake_endpoint", aws_profile_name="invalid-profile"
    )
