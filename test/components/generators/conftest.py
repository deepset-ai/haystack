from datetime import datetime
from typing import Iterator
from unittest.mock import patch, MagicMock

import pytest
from openai import Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice


@pytest.fixture
def mock_auto_tokenizer():
    """
    In the original mock_auto_tokenizer fixture, we were mocking the transformers.AutoTokenizer.from_pretrained
    method directly, but we were not providing a return value for this method. Therefore, when from_pretrained
    was called within HuggingFaceTGIChatGenerator, it returned None because that's the default behavior of a
    MagicMock object when a return value isn't specified.

    We will update the mock_auto_tokenizer fixture to return a MagicMock object when from_pretrained is called
    in another PR. For now, we will use this fixture to mock the AutoTokenizer.from_pretrained method.
    """

    with patch("transformers.AutoTokenizer.from_pretrained", autospec=True) as mock_from_pretrained:
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        yield mock_tokenizer


@pytest.fixture
def mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    class MockStream(Stream[ChatCompletionChunk]):
        def __init__(self, mock_chunk: ChatCompletionChunk, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mock_chunk = mock_chunk

        def __stream__(self) -> Iterator[ChatCompletionChunk]:
            # Yielding only one ChatCompletionChunk object
            yield self.mock_chunk

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    finish_reason="stop", logprobs=None, index=0, delta=ChoiceDelta(content="Hello", role="assistant")
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = MockStream(completion, cast_to=None, response=None, client=None)
        yield mock_chat_completion_create
