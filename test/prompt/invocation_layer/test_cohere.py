import unittest
from unittest.mock import patch, MagicMock

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import CohereInvocationLayer


@pytest.mark.unit
def test_default_constructor():
    """
    Test that the default constructor sets the correct values
    """

    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key")

    assert layer.api_key == "some_fake_key"
    assert layer.max_length == 100
    assert layer.model_input_kwargs == {}
    assert layer.prompt_handler.model_max_length == 4096

    layer = CohereInvocationLayer(model_name_or_path="base", api_key="some_fake_key")
    assert layer.api_key == "some_fake_key"
    assert layer.max_length == 100
    assert layer.model_input_kwargs == {}
    assert layer.prompt_handler.model_max_length == 2048


@pytest.mark.unit
def test_constructor_with_model_kwargs():
    """
    Test that model_kwargs are correctly set in the constructor
    and that model_kwargs_rejected are correctly filtered out
    """
    model_kwargs = {"temperature": 0.7, "end_sequences": ["end"], "stream": True}
    model_kwargs_rejected = {"fake_param": 0.7, "another_fake_param": 1}
    layer = CohereInvocationLayer(
        model_name_or_path="command", api_key="some_fake_key", **model_kwargs, **model_kwargs_rejected
    )
    assert layer.model_input_kwargs == model_kwargs
    assert len(model_kwargs_rejected) == 2


@pytest.mark.unit
def test_invoke_with_no_kwargs():
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key")
    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match("No prompt provided.")


@pytest.mark.unit
def test_invoke_with_stop_words():
    """
    Test stop words are correctly passed from PromptNode to wire in CohereInvocationLayer
    """
    stop_words = ["but", "not", "bye"]
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")
    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='{"generations":[{"text": "Hello"}]}')

        layer.invoke(prompt="Tell me hello", stop_words=stop_words)

        assert mock_post.called

        # Check if stop_words are passed to _post as stop parameter
        called_args, _ = mock_post.call_args
        assert "end_sequences" in called_args[0]
        assert called_args[0]["end_sequences"] == stop_words


@pytest.mark.unit
@pytest.mark.parametrize("using_constructor", [True, False])
@pytest.mark.parametrize("stream", [True, False])
def test_streaming_stream_param(using_constructor, stream):
    """
    Test stream parameter is correctly passed from PromptNode to wire in CohereInvocationLayer
    """
    if using_constructor:
        layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key", stream=stream)
    else:
        layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='{"generations":[{"text": "Hello"}]}')

        if using_constructor:
            layer.invoke(prompt="Tell me hello")
        else:
            layer.invoke(prompt="Tell me hello", stream=stream)

        assert mock_post.called

        # Check if stop_words are passed to _post as stop parameter
        called_args, called_kwargs = mock_post.call_args

        # stream is always passed to _post
        assert "stream" in called_kwargs

        # Check if stream is True, then stream is passed as True to _post
        if stream:
            assert called_kwargs["stream"]
        # Check if stream is False, then stream is passed as False to _post
        else:
            assert not called_kwargs["stream"]


@pytest.mark.unit
@pytest.mark.parametrize("using_constructor", [True, False])
@pytest.mark.parametrize("stream_handler", [DefaultTokenStreamingHandler(), None])
def test_streaming_stream_handler_param(using_constructor, stream_handler):
    """
    Test stream_handler parameter is correctly from PromptNode passed to wire in CohereInvocationLayer
    """
    if using_constructor:
        layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key", stream_handler=stream_handler)
    else:
        layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")

    with unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post"
    ) as mock_post, unittest.mock.patch(
        "haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._process_streaming_response"
    ) as mock_post_stream:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = MagicMock(text='{"generations":[{"text": "Hello"}]}')

        if using_constructor:
            layer.invoke(prompt="Tell me hello")
        else:
            layer.invoke(prompt="Tell me hello", stream_handler=stream_handler)

        assert mock_post.called

        # Check if stop_words are passed to _post as stop parameter
        called_args, called_kwargs = mock_post.call_args

        # stream is always passed to _post
        assert "stream" in called_kwargs

        # if stream_handler is used then stream is always True
        if stream_handler:
            assert called_kwargs["stream"]
            # and stream_handler is passed as an instance of TokenStreamingHandler
            called_args, called_kwargs = mock_post_stream.call_args
            assert "stream_handler" in called_kwargs
            assert isinstance(called_kwargs["stream_handler"], TokenStreamingHandler)
        # if stream_handler is not used then stream is always False
        else:
            assert not called_kwargs["stream"]


@pytest.mark.unit
def test_supports():
    """
    Test that supports returns True correctly for CohereInvocationLayer
    """
    # See command and generate models at https://docs.cohere.com/docs/models
    # doesn't support fake model
    assert not CohereInvocationLayer.supports("fake_model", api_key="fake_key")

    # supports cohere command with api_key
    assert CohereInvocationLayer.supports("command", api_key="fake_key")

    # supports cohere command-light with api_key
    assert CohereInvocationLayer.supports("command-light", api_key="fake_key")

    # supports cohere base with api_key
    assert CohereInvocationLayer.supports("base", api_key="fake_key")

    assert CohereInvocationLayer.supports("base-light", api_key="fake_key")

    # doesn't support other models that have base substring only i.e. google/flan-t5-base
    assert not CohereInvocationLayer.supports("google/flan-t5-base")


@pytest.mark.unit
def test_ensure_token_limit_fails_if_called_with_list():
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key")
    with pytest.raises(ValueError):
        layer._ensure_token_limit(prompt=[])


@pytest.mark.unit
def test_ensure_token_limit_with_small_max_length(caplog):
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key", max_length=10)
    res = layer._ensure_token_limit(prompt="Short prompt")

    assert res == "Short prompt"
    assert not caplog.records

    res = layer._ensure_token_limit(prompt="This is a very very very very very much longer prompt")
    assert res == "This is a very very very very very much longer prompt"
    assert not caplog.records


@pytest.mark.unit
def test_ensure_token_limit_with_huge_max_length(caplog):
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key", max_length=4090)
    res = layer._ensure_token_limit(prompt="Short prompt")

    assert res == "Short prompt"
    assert not caplog.records

    res = layer._ensure_token_limit(prompt="This is a very very very very very much longer prompt")
    assert res == "This is a very very very"
    assert len(caplog.records) == 1
    expected_message_log = (
        "The prompt has been truncated from 11 tokens to 6 tokens so that the prompt length and "
        "answer length (4090 tokens) fit within the max token limit (4096 tokens). "
        "Reduce the length of the prompt to prevent it from being cut off."
    )
    assert caplog.records[0].message == expected_message_log
