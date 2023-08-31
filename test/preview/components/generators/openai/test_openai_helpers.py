from unittest.mock import Mock, patch
import json

import pytest


from haystack.preview.components.generators.openai.errors import (
    OpenAIUnauthorizedError,
    OpenAIError,
    OpenAIRateLimitError,
)
from haystack.preview.components.generators.openai._helpers import (
    raise_for_status,
    check_truncated_answers,
    check_filtered_answers,
    query_chat_model,
    query_chat_model_stream,
    enforce_token_limit,
    enforce_token_limit_chat,
    OPENAI_TIMEOUT,
    OPENAI_MAX_RETRIES,
)


@pytest.mark.unit
def test_raise_for_status_200():
    response = Mock()
    response.status_code = 200
    raise_for_status(response)


@pytest.mark.unit
def test_raise_for_status_401():
    response = Mock()
    response.status_code = 401
    with pytest.raises(OpenAIUnauthorizedError):
        raise_for_status(response)


@pytest.mark.unit
def test_raise_for_status_429():
    response = Mock()
    response.status_code = 429
    with pytest.raises(OpenAIRateLimitError):
        raise_for_status(response)


@pytest.mark.unit
def test_raise_for_status_500():
    response = Mock()
    response.status_code = 500
    response.text = "Internal Server Error"
    with pytest.raises(OpenAIError):
        raise_for_status(response)


@pytest.mark.unit
def test_check_truncated_answers(caplog):
    result = {
        "choices": [
            {"finish_reason": "length"},
            {"finish_reason": "content_filter"},
            {"finish_reason": "length"},
            {"finish_reason": "stop"},
        ]
    }
    payload = {"n": 4}
    check_truncated_answers(result, payload)
    assert caplog.records[0].message == (
        "2 out of the 4 completions have been truncated before reaching a natural "
        "stopping point. Increase the max_tokens parameter to allow for longer completions."
    )


@pytest.mark.unit
def test_check_truncated_answers(caplog):
    result = {"choices": [{"finish_reason": "content_filter"}, {"finish_reason": "length"}, {"finish_reason": "stop"}]}
    payload = {"n": 3}
    check_filtered_answers(result, payload)
    assert caplog.records[0].message == (
        "1 out of the 3 completions have omitted content due to a flag from OpenAI content filters."
    )


@pytest.mark.unit
def test_query_chat_model():
    with patch("haystack.preview.components.generators.openai._helpers.requests.post") as mock_post:
        response = Mock()
        response.status_code = 200
        response.text = '{"choices": [{"finish_reason": "stop", "message": {"content": "   Hello, how are you? "}}]}'
        mock_post.return_value = response
        replies = query_chat_model(url="test-url", headers={"header": "test-header"}, payload={"param": "test-param"})
        mock_post.assert_called_once_with(
            "test-url",
            headers={"header": "test-header"},
            data=json.dumps({"param": "test-param"}),
            timeout=OPENAI_TIMEOUT,
        )
        assert replies == ["Hello, how are you?"]


@pytest.mark.unit
def test_query_chat_model_fail():
    with patch("haystack.preview.components.generators.openai._helpers.requests.post") as mock_post:
        response = Mock()
        response.status_code = 500
        mock_post.return_value = response
        with pytest.raises(OpenAIError):
            query_chat_model(url="test-url", headers={"header": "test-header"}, payload={"param": "test-param"})
            mock_post.assert_called_with(
                "test-url",
                headers={"header": "test-header"},
                data=json.dumps({"param": "test-param"}),
                timeout=OPENAI_TIMEOUT,
            )
            mock_post.call_count == OPENAI_MAX_RETRIES


@pytest.mark.unit
def test_query_chat_model_stream():
    with patch("haystack.preview.components.generators.openai._helpers.requests.post") as mock_post:
        with patch("haystack.preview.components.generators.openai._helpers.sseclient.SSEClient") as mock_sseclient:
            callback = lambda token, event_data: f"|{token}|"
            response = Mock()
            response.status_code = 200

            mock_sseclient.return_value.events.return_value = [
                Mock(data='{"choices": [{"delta": {"content": "Hello"}}]}'),
                Mock(data='{"choices": [{"delta": {"content": ","}}]}'),
                Mock(data='{"choices": [{"delta": {"content": " how"}}]}'),
                Mock(data='{"choices": [{"delta": {"content": " are"}}]}'),
                Mock(data='{"choices": [{"delta": {"content": " you"}}]}'),
                Mock(data='{"choices": [{"delta": {"content": "?"}}]}'),
                Mock(data="test-marker"),
                Mock(data='{"choices": [{"delta": {"content": "discarded tokens"}}]}'),
            ]

            mock_post.return_value = response
            replies = query_chat_model_stream(
                url="test-url",
                headers={"header": "test-header"},
                payload={"param": "test-param"},
                callback=callback,
                marker="test-marker",
            )
            mock_post.assert_called_once_with(
                "test-url",
                headers={"header": "test-header"},
                data=json.dumps({"param": "test-param"}),
                timeout=OPENAI_TIMEOUT,
            )
            assert replies == ["|Hello||,|| how|| are|| you||?|"]


@pytest.mark.unit
def test_query_chat_model_stream_fail():
    with patch("haystack.preview.components.generators.openai._helpers.requests.post") as mock_post:
        callback = Mock()
        response = Mock()
        response.status_code = 500
        mock_post.return_value = response
        with pytest.raises(OpenAIError):
            query_chat_model_stream(
                url="test-url",
                headers={"header": "test-header"},
                payload={"param": "test-param"},
                callback=callback,
                marker="test-marker",
            )
            mock_post.assert_called_with(
                "test-url",
                headers={"header": "test-header"},
                data=json.dumps({"param": "test-param"}),
                timeout=OPENAI_TIMEOUT,
            )
            mock_post.call_count == OPENAI_MAX_RETRIES


@pytest.mark.unit
def test_enforce_token_limit_above_limit(caplog, mock_tokenizer):
    prompt = enforce_token_limit("This is a test prompt.", tokenizer=mock_tokenizer, max_tokens_limit=3)
    assert prompt == "This is a"
    assert caplog.records[0].message == (
        "The prompt has been truncated from 5 tokens to 3 tokens to fit within the max token "
        "limit. Reduce the length of the prompt to prevent it from being cut off."
    )


@pytest.mark.unit
def test_enforce_token_limit_below_limit(caplog, mock_tokenizer):
    prompt = enforce_token_limit("This is a test prompt.", tokenizer=mock_tokenizer, max_tokens_limit=100)
    assert prompt == "This is a test prompt."
    assert not caplog.records


@pytest.mark.unit
def test_enforce_token_limit_chat_above_limit(caplog, mock_tokenizer):
    prompts = enforce_token_limit_chat(
        ["System Prompt", "This is a test prompt."],
        tokenizer=mock_tokenizer,
        max_tokens_limit=7,
        tokens_per_message_overhead=2,
    )
    assert prompts == ["System Prompt", "This is a"]
    assert caplog.records[0].message == (
        "The prompts have been truncated from 11 tokens to 7 tokens to fit within the max token limit. "
        "Reduce the length of the prompt to prevent it from being cut off."
    )


@pytest.mark.unit
def test_enforce_token_limit_chat_below_limit(caplog, mock_tokenizer):
    prompts = enforce_token_limit_chat(
        ["System Prompt", "This is a test prompt."],
        tokenizer=mock_tokenizer,
        max_tokens_limit=100,
        tokens_per_message_overhead=2,
    )
    assert prompts == ["System Prompt", "This is a test prompt."]
    assert not caplog.records
