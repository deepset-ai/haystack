import pytest
import logging
import unittest
from unittest.mock import patch, MagicMock, Mock

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import SageMakerInvocationLayer


@pytest.mark.integration
def test_supports():
    """
    Test that supports returns True for valid SageMakerInvocationLayer
    """

    assert SageMakerInvocationLayer.supports(
        model_name_or_path="jumpstart-example-tiiuae-falcon-40b-ins-2023-06-16-09-15-35-027",
        profile_name="Haystack-OSS-test",
    )


@pytest.mark.integration
def test_supports_not():
    """
    Test that supports returns False for invalid SageMakerInvocationLayer
    """
    assert not SageMakerInvocationLayer.supports("google/flan-t5-xxl", profile_name="Haystack-OSS-test")
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="jumpstart-example-tiiuae-falcon-40b-ins-2023-06-16-09-15-35-027"
    )
    assert not SageMakerInvocationLayer.supports(
        model_name_or_path="invalid-model-name", profile_name="invalid-profile"
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

    layer = SageMakerInvocationLayer(model_name_or_path="flan-t5-xxl", max_length=99, aws_access_key_id="some_fake_id", aws_secret_access_key="some_fake_key", aws_session_token="some_fake_token", profile_name="some_fake_profile", region_name="fake_region")

    # assert layer. == "some_fake_key"
    assert layer.max_length == 99
    assert layer.model_name_or_path == "flan-t5-xxl"

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
def test_constructor_with_model_kwargs(mock_auto_tokenizer):
    """
    Test that model_kwargs are correctly set in the constructor
    and that model_kwargs_rejected are correctly filtered out
    """
    model_kwargs = {"temperature": 0.7, "do_sample": True, "stream": True}
    model_kwargs_rejected = {"fake_param": 0.7, "another_fake_param": 1}

    layer = HFInferenceEndpointInvocationLayer(
        model_name_or_path="google/flan-t5-xxl", api_key="some_fake_key", **model_kwargs, **model_kwargs_rejected
    )
    assert "temperature" in layer.model_input_kwargs
    assert "do_sample" in layer.model_input_kwargs
    assert "stream" in layer.model_input_kwargs
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





@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name_or_path", ["google/flan-t5-xxl", "OpenAssistant/oasst-sft-1-pythia-12b", "bigscience/bloomz"]
)
def test_ensure_token_limit_resize(caplog, model_name_or_path):
    # In this test case we assume prompt resizing is needed for all models
    handler = HFInferenceEndpointInvocationLayer("fake_api_key", model_name_or_path, max_length=5, model_max_length=10)

    # Define prompt and expected results
    prompt = "This is a test prompt that will be resized because model_max_length is 10 and max_length is 5."
    with caplog.at_level(logging.WARN):
        resized_prompt = handler._ensure_token_limit(prompt)
        assert "The prompt has been truncated" in caplog.text

    # Verify the results
    assert resized_prompt != prompt
    assert (
        "This is a test" in resized_prompt
        and "because model_max_length is 10 and max_length is 5" not in resized_prompt
    )


@pytest.mark.unit
def test_oasst_prompt_preprocessing(mock_auto_tokenizer):
    model_name = "OpenAssistant/oasst-sft-1-pythia-12b"

    layer = HFInferenceEndpointInvocationLayer("fake_api_key", model_name)
    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        result = layer.invoke(prompt="Tell me hello")

    assert result == ["Hello"]
    assert mock_post.called

    _, called_kwargs = mock_post.call_args
    # OpenAssistant/oasst-sft-1-pythia-12b prompts are preprocessed and wrapped in tokens below
    assert called_kwargs["data"]["inputs"] == "<|prompter|>Tell me hello<|endoftext|><|assistant|>"


@pytest.mark.unit
def test_invalid_key():
    with pytest.raises(ValueError, match="must be a valid Hugging Face token"):
        HFInferenceEndpointInvocationLayer("", "irrelevant_model_name")


@pytest.mark.unit
def test_invalid_model():
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        HFInferenceEndpointInvocationLayer("fake_api", "")


@pytest.mark.unit
def test_supports(mock_get_task):
    """
    Test that supports returns True correctly for HFInferenceEndpointInvocationLayer
    """

    # supports google/flan-t5-xxl with api_key
    assert HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl", api_key="fake_key")

    # doesn't support google/flan-t5-xxl without api_key
    assert not HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl")

    # supports HF Inference Endpoint with api_key
    assert HFInferenceEndpointInvocationLayer.supports(
        "https://<your-unique-deployment-id>.us-east-1.aws.endpoints.huggingface.cloud", api_key="fake_key"
    )


@pytest.mark.unit
def test_supports_not(mock_get_task_invalid):
    assert not HFInferenceEndpointInvocationLayer.supports("fake_model", api_key="fake_key")

