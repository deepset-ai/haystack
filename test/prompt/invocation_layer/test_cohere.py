import unittest
from unittest.mock import Mock

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import CohereInvocationLayer


@pytest.mark.unit
def test_default_constructor(mock_auto_tokenizer):
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
def test_constructor_with_model_kwargs(mock_auto_tokenizer):
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
def test_invoke_with_no_kwargs(mock_auto_tokenizer):
    """
    Test that invoke raises an error if no prompt is provided
    """
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key")
    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match("No prompt provided.")


@pytest.mark.unit
def test_invoke_with_stop_words(mock_auto_tokenizer):
    """
    Test stop words are correctly passed from PromptNode to wire in CohereInvocationLayer
    """
    stop_words = ["but", "not", "bye"]
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")
    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response, need to return a list of dicts
        mock_post.return_value = Mock(text='{"generations":[{"text": "Hello"}]}')

        layer.invoke(prompt="Tell me hello", stop_words=stop_words)

    assert mock_post.called
    called_args, _ = mock_post.call_args
    assert "end_sequences" in called_args[0]
    assert called_args[0]["end_sequences"] == stop_words


@pytest.mark.unit
def test_streaming_stream_param_from_init(mock_auto_tokenizer):
    """
    Test stream parameter is correctly passed from PromptNode to wire in CohereInvocationLayer from init
    """

    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key", stream=True)

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(iter_lines=Mock(return_value=['{"text": "Hello"}', '{"text": " there"}']))
        layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs and called_kwargs["stream"]


@pytest.mark.unit
def test_streaming_stream_param_from_init_no_stream(mock_auto_tokenizer):
    """
    Test stream parameter is correctly passed from PromptNode to wire in CohereInvocationLayer from init
    """

    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(text='{"generations":[{"text": "Hello there"}]}')
        layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs
    assert not bool(called_kwargs["stream"])


@pytest.mark.unit
def test_streaming_stream_param_from_invoke(mock_auto_tokenizer):
    """
    Test stream parameter is correctly passed from PromptNode to wire in CohereInvocationLayer from invoke
    """
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(iter_lines=Mock(return_value=['{"text": "Hello"}', '{"text": " there"}']))
        layer.invoke(prompt="Tell me hello", stream=True)

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs
    assert bool(called_kwargs["stream"])


@pytest.mark.unit
def test_streaming_stream_param_from_invoke_no_stream(mock_auto_tokenizer):
    """
    Test stream parameter is correctly passed from PromptNode to wire in CohereInvocationLayer from invoke
    """
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key", stream=True)

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(text='{"generations":[{"text": "Hello there"}]}')
        layer.invoke(prompt="Tell me hello", stream=False)

    assert mock_post.called
    _, called_kwargs = mock_post.call_args

    # stream is always passed to _post
    assert "stream" in called_kwargs
    assert not bool(called_kwargs["stream"])


@pytest.mark.unit
def test_streaming_stream_handler_param_from_init(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly from PromptNode passed to wire in CohereInvocationLayer
    """
    stream_handler = DefaultTokenStreamingHandler()
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key", stream_handler=stream_handler)

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(iter_lines=Mock(return_value=['{"text": "Hello"}', '{"text": " there"}']))
        responses = layer.invoke(prompt="Tell me hello")

    assert mock_post.called
    _, called_kwargs = mock_post.call_args
    assert "stream" in called_kwargs
    assert bool(called_kwargs["stream"])
    assert responses == ["Hello there"]


@pytest.mark.unit
def test_streaming_stream_handler_param_from_invoke(mock_auto_tokenizer):
    """
    Test stream_handler parameter is correctly from PromptNode passed to wire in CohereInvocationLayer
    """
    stream_handler = DefaultTokenStreamingHandler()
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="fake_key")

    with unittest.mock.patch("haystack.nodes.prompt.invocation_layer.CohereInvocationLayer._post") as mock_post:
        # Mock the response
        mock_post.return_value = Mock(iter_lines=Mock(return_value=['{"text": "Hello"}', '{"text": " there"}']))
        responses = layer.invoke(prompt="Tell me hello", stream_handler=stream_handler)

    assert mock_post.called
    _, called_kwargs = mock_post.call_args
    assert "stream" in called_kwargs
    assert bool(called_kwargs["stream"])
    assert responses == ["Hello there"]


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
def test_ensure_token_limit_fails_if_called_with_list(mock_auto_tokenizer):
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key")
    with pytest.raises(ValueError):
        layer._ensure_token_limit(prompt=[])


@pytest.mark.integration
def test_ensure_token_limit_with_small_max_length(caplog):
    layer = CohereInvocationLayer(model_name_or_path="command", api_key="some_fake_key", max_length=10)
    res = layer._ensure_token_limit(prompt="Short prompt")

    assert res == "Short prompt"
    assert not caplog.records

    res = layer._ensure_token_limit(prompt="This is a very very very very very much longer prompt")
    assert res == "This is a very very very very very much longer prompt"
    assert not caplog.records


@pytest.mark.integration
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
