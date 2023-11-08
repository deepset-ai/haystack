from unittest.mock import patch, MagicMock

import pytest

from haystack.lazy_imports import LazyImport

from haystack.errors import AmazonBedrockConfigurationError
from haystack.nodes.prompt.invocation_layer import AmazonBedrockBaseInvocationLayer

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

    layer = AmazonBedrockBaseInvocationLayer(
        model_name_or_path="anthropic.claude-v2",
        max_length=99,
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        aws_profile_name="some_fake_profile",
        aws_region_name="fake_region",
    )

    assert layer.max_length == 99
    assert layer.model_name_or_path == "anthropic.claude-v2"

    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096

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
def test_constructor_prompt_handler_initialized(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that the constructor sets the prompt_handler correctly, with the correct model_max_length for llama-2
    """
    layer = AmazonBedrockBaseInvocationLayer(
        model_name_or_path="anthropic.claude-v2", prompt_handler=mock_prompt_handler
    )
    assert layer.prompt_handler is not None
    assert layer.prompt_handler.model_max_length == 4096


@pytest.mark.unit
def test_constructor_with_model_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test that model_kwargs are correctly set in the constructor
    """
    model_kwargs = {"temperature": 0.7}

    layer = AmazonBedrockBaseInvocationLayer(model_name_or_path="anthropic.claude-v2", **model_kwargs)
    assert "temperature" in layer.model_input_kwargs
    assert layer.model_input_kwargs["temperature"] == 0.7


@pytest.mark.unit
def test_constructor_with_empty_model_name():
    """
    Test that the constructor raises an error when the model_name_or_path is empty
    """
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockBaseInvocationLayer(model_name_or_path="")


@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer, mock_boto3_session):
    """
    Test invoke raises an error if no prompt is provided
    """
    layer = AmazonBedrockBaseInvocationLayer(model_name_or_path="anthropic.claude-v2")
    with pytest.raises(ValueError, match="No valid prompt provided."):
        layer.invoke()


@pytest.mark.unit
def test_short_prompt_is_not_truncated(mock_boto3_session):
    """
    Test that a short prompt is not truncated
    """
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
        layer = AmazonBedrockBaseInvocationLayer(
            "anthropic.claude-v2", max_length=max_length_generated_text, model_max_length=total_model_max_length
        )
        prompt_after_resize = layer._ensure_token_limit(mock_prompt_text)

    # The prompt doesn't exceed the limit, _ensure_token_limit doesn't truncate it
    assert prompt_after_resize == mock_prompt_text


@pytest.mark.unit
def test_long_prompt_is_truncated(mock_boto3_session):
    """
    Test that a long prompt is truncated
    """
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
        layer = AmazonBedrockBaseInvocationLayer(
            "anthropic.claude-v2", max_length=max_length_generated_text, model_max_length=total_model_max_length
        )
        prompt_after_resize = layer._ensure_token_limit(long_prompt_text)

    # The prompt exceeds the limit, _ensure_token_limit truncates it
    assert prompt_after_resize == truncated_prompt_text


@pytest.mark.unit
def test_supports_for_valid_aws_configuration():
    mock_session = MagicMock()
    mock_session.client("bedrock").list_foundation_models.return_value = {
        "modelSummaries": [{"modelId": "anthropic.claude-v2"}]
    }

    # Patch the class method to return the mock session
    with patch(
        "haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session",
        return_value=mock_session,
    ):
        supported = AmazonBedrockBaseInvocationLayer.supports(
            model_name_or_path="anthropic.claude-v2", aws_profile_name="some_real_profile"
        )
    args, kwargs = mock_session.client("bedrock").list_foundation_models.call_args
    assert kwargs["byOutputModality"] == "TEXT"

    assert supported


@pytest.mark.unit
def test_supports_raises_on_invalid_aws_profile_name():
    with patch("boto3.Session") as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(AmazonBedrockConfigurationError, match="Failed to initialize the session"):
            AmazonBedrockBaseInvocationLayer.supports(
                model_name_or_path="anthropic.claude-v2", aws_profile_name="some_fake_profile"
            )


@pytest.mark.unit
def test_supports_for_invalid_bedrock_config():
    mock_session = MagicMock()
    mock_session.client.side_effect = BotoCoreError()

    # Patch the class method to return the mock session
    with patch(
        "haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session",
        return_value=mock_session,
    ), pytest.raises(AmazonBedrockConfigurationError, match="Could not connect to Amazon Bedrock."):
        AmazonBedrockBaseInvocationLayer.supports(
            model_name_or_path="anthropic.claude-v2", aws_profile_name="some_real_profile"
        )


@pytest.mark.unit
def test_supports_for_invalid_bedrock_config_error_on_list_models():
    mock_session = MagicMock()
    mock_session.client("bedrock").list_foundation_models.side_effect = BotoCoreError()

    # Patch the class method to return the mock session
    with patch(
        "haystack.nodes.prompt.invocation_layer.aws_base.AWSBaseInvocationLayer.get_aws_session",
        return_value=mock_session,
    ), pytest.raises(AmazonBedrockConfigurationError, match="Could not connect to Amazon Bedrock."):
        AmazonBedrockBaseInvocationLayer.supports(
            model_name_or_path="anthropic.claude-v2", aws_profile_name="some_real_profile"
        )


@pytest.mark.unit
def test_supports_for_no_aws_params():
    supported = AmazonBedrockBaseInvocationLayer.supports(model_name_or_path="anthropic.claude-v2")

    assert supported == False


@pytest.mark.unit
def test_supports_for_unknown_model():
    supported = AmazonBedrockBaseInvocationLayer.supports(
        model_name_or_path="unknown_model", aws_profile_name="some_real_profile"
    )

    assert supported == False
