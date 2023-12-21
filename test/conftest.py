from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture()
def mock_tokenizer():
    """
    Tokenizes the string by splitting on spaces.
    """
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)
    return tokenizer


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture(autouse=True)
def request_blocker(request: pytest.FixtureRequest, monkeypatch):
    """
    This fixture is applied automatically to all tests.
    Those that are not marked as integration will have the requests module
    monkeypatched to avoid making HTTP requests by mistake.
    """
    marker = request.node.get_closest_marker("integration")
    if marker is not None:
        return

    def urlopen_mock(self, method, url, *args, **kwargs):
        raise RuntimeError(f"The test was about to {method} {self.scheme}://{self.host}{url}")

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen_mock)
