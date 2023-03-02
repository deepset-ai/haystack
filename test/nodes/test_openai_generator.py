import logging
import time

import pytest
import requests

from haystack.errors import OpenAIError
from haystack.nodes.answer_generator import OpenAIAnswerGenerator


class MockResponse:
    def __init__(self, text, status_code):
        self.text = text
        self.status_code = status_code


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    # Quite harsh, but 'retry_with_exponential_backoff' is resisting to my mocking attempts. FIXME
    monkeypatch.setattr(time, "sleep", lambda func: func)


@pytest.fixture
def mock_requests(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: MockResponse(
            text='{"choices": [{"text": "MOCK ANSWER", "finish_reason": "not-length"}]}', status_code=200
        ),
    )


@pytest.fixture
def openai_generator(mock_requests):
    return OpenAIAnswerGenerator(api_key="irrelevant-anyway", top_k=1)


@pytest.mark.unit
def test_openai_answer_generator(openai_generator, docs):
    prediction = openai_generator.predict(query="Test query", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "MOCK ANSWER" == prediction["answers"][0].answer


@pytest.mark.unit
def test_openai_answer_generator_server_error(monkeypatch, docs):
    monkeypatch.setattr(
        requests, "request", lambda *a, **k: MockResponse(text='{"error": "testing errors"}', status_code=500)
    )

    openai_generator = OpenAIAnswerGenerator(api_key="irrelevant-anyway", top_k=1)
    with pytest.raises(OpenAIError):
        openai_generator.predict(query="Test query", documents=docs, top_k=1)


@pytest.mark.unit
def test_openai_answer_generator_rate_limit(monkeypatch, docs):
    monkeypatch.setattr(
        requests,
        "request",
        lambda *a, **k: MockResponse(text='{"error": "testing rate limit errors"}', status_code=429),
    )
    openai_generator = OpenAIAnswerGenerator(api_key="irrelevant-anyway", top_k=1)
    with pytest.raises(OpenAIError):
        openai_generator.predict(query="Test query", documents=docs, top_k=1)


@pytest.mark.unit
def test_openai_answer_generator_max_token(openai_generator, docs, caplog):
    openai_generator.MAX_TOKENS_LIMIT = 116
    with caplog.at_level(logging.INFO):
        prediction = openai_generator.predict(query="Test query", documents=docs, top_k=1)
        assert "Skipping all of the provided Documents" in caplog.text
        assert len(prediction["answers"]) == 1
