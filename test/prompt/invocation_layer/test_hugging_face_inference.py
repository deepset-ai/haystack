import logging
import unittest
from unittest.mock import patch, MagicMock

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import HFInferenceEndpointInvocationLayer


@pytest.fixture
def mock_get_task():
    # mock get_task function
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face_inference.get_task") as mock_get_task:
        mock_get_task.return_value = "text2text-generation"
        yield mock_get_task


@pytest.fixture
def mock_get_task_invalid():
    with patch("haystack.nodes.prompt.invocation_layer.hugging_face_inference.get_task") as mock_get_task:
        mock_get_task.return_value = "some-nonexistent-type"
        yield mock_get_task


@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer):
    """
    Test that the default constructor sets the correct values
    """

    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="some_fake_key")

    assert layer.api_key == "some_fake_key"
    assert layer.max_length == 100
    assert layer.model_input_kwargs == {}


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
def test_set_model_max_length(mock_auto_tokenizer):
    """
    Test that model max length is set correctly
    """
    layer = HFInferenceEndpointInvocationLayer(
        model_name_or_path="google/flan-t5-xxl", api_key="some_fake_key", model_max_length=2048
    )
    assert layer.prompt_handler.model_max_length == 2048


@pytest.mark.unit
def test_url(mock_auto_tokenizer):
    """
    Test that the url is correctly set in the constructor
    """
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="some_fake_key")
    assert layer.url == "https://api-inference.huggingface.co/models/google/flan-t5-xxl"

    layer = HFInferenceEndpointInvocationLayer(
        model_name_or_path="https://23445.us-east-1.aws.endpoints.huggingface.cloud", api_key="some_fake_key"
    )

    assert layer.url == "https://23445.us-east-1.aws.endpoints.huggingface.cloud"


@pytest.mark.unit
def test_invoke_with_no_kwargs(mock_auto_tokenizer):
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="some_fake_key")
    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match("No prompt provided.")


@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer):
    """
    Test stop words are correctly passed to HTTP POST request
    """
    stop_words = ["but", "not", "bye"]
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="fake_key")
    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')

        layer.invoke(prompt="Tell me hello", stop_words=stop_words)

    assert mock_post.called

    # Check if stop_words are passed to _post as stop parameter
    _, called_kwargs = mock_post.call_args
    assert "stop" in called_kwargs["data"]["parameters"]
    assert called_kwargs["data"]["parameters"]["stop"] == stop_words


@pytest.mark.unit
@pytest.mark.parametrize("stream", [True, False])
def test_streaming_stream_param_in_constructor(mock_auto_tokenizer, stream):
    """
    Test stream parameter is correctly passed to HTTP POST request via constructor
    """
    layer = HFInferenceEndpointInvocationLayer(
        model_name_or_path="google/flan-t5-xxl", api_key="fake_key", stream=stream
    )
    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs

    assert called_kwargs["stream"] == stream


@pytest.mark.unit
@pytest.mark.parametrize("stream", [True, False])
def test_streaming_stream_param_in_method(mock_auto_tokenizer, stream):
    """
    Test stream parameter is correctly passed to HTTP POST request via method
    """
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="fake_key")
    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt="Tell me hello", stream=stream)

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs

    # Assert that the 'stream' parameter passed to _post is the same as the one used in layer.invoke()
    assert called_kwargs["stream"] == stream


@pytest.mark.unit
def test_streaming_stream_handler_param_in_constructor(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly passed to HTTP POST request via constructor
    """
    stream_handler = DefaultTokenStreamingHandler()
    layer = HFInferenceEndpointInvocationLayer(
        model_name_or_path="google/flan-t5-xxl", api_key="fake_key", stream_handler=stream_handler
    )

    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post, unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._process_streaming_response"
    ) as mock_post_stream:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs

    assert called_kwargs["stream"]

    # stream_handler is passed as an instance of TokenStreamingHandler
    called_args, _ = mock_post_stream.call_args
    assert isinstance(called_args[1], TokenStreamingHandler)


@pytest.mark.unit
def test_streaming_no_stream_handler_param_in_constructor(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly passed to HTTP POST request via constructor
    """
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="fake_key")

    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')
        layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs

    # but it is False if stream_handler is None
    assert not called_kwargs["stream"]


@pytest.mark.unit
def test_streaming_stream_handler_param_in_method(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly passed to HTTP POST request via method
    """
    stream_handler = DefaultTokenStreamingHandler()
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="fake_key")

    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post, unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._process_streaming_response"
    ) as mock_post_stream:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')

        layer.invoke(prompt="Tell me hello", stream_handler=stream_handler)

    assert mock_post.called
    called_args, called_kwargs = mock_post.call_args

    # stream is correctly passed to _post
    assert "stream" in called_kwargs
    assert called_kwargs["stream"]

    called_args, called_kwargs = mock_post_stream.call_args
    assert isinstance(called_args[1], TokenStreamingHandler)


@pytest.mark.unit
def test_streaming_no_stream_handler_param_in_method(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly passed to HTTP POST request via method
    """
    layer = HFInferenceEndpointInvocationLayer(model_name_or_path="google/flan-t5-xxl", api_key="fake_key")

    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.HFInferenceEndpointInvocationLayer._post"
    ) as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='[{"generated_text": "Hello"}]')

        layer.invoke(prompt="Tell me hello", stream_handler=None)

    assert mock_post.called

    _, called_kwargs = mock_post.call_args

    # stream is always correctly passed to _post
    assert "stream" in called_kwargs
    assert not called_kwargs["stream"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name_or_path", ["google/flan-t5-xxl", "OpenAssistant/oasst-sft-1-pythia-12b", "bigscience/bloomz"]
)
def test_ensure_token_limit_no_resize(model_name_or_path):
    # In this test case we assume that no prompt resizing is needed for all models
    handler = HFInferenceEndpointInvocationLayer("fake_api_key", model_name_or_path, max_length=100)

    # Define prompt and expected results
    prompt = "This is a test prompt."

    resized_prompt = handler._ensure_token_limit(prompt)

    # Verify the results
    assert resized_prompt == prompt


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

    # supports HF Inference Endpoint with api_key
    assert HFInferenceEndpointInvocationLayer.supports(
        "https://<your-unique-deployment-id>.us-east-1.aws.endpoints.huggingface.cloud", api_key="fake_key"
    )


@pytest.mark.unit
def test_supports_not(mock_get_task_invalid):
    assert not HFInferenceEndpointInvocationLayer.supports("fake_model", api_key="fake_key")

    # doesn't support google/flan-t5-xxl without api_key
    assert not HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl")

    # doesn't support HF Inference Endpoint without proper api_key
    assert not HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl", api_key="")
    assert not HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl", api_key=None)

    # doesn't support model if hf server timeout
    mock_get_task.side_effect = RuntimeError
    assert not HFInferenceEndpointInvocationLayer.supports("google/flan-t5-xxl", api_key="fake_key")
