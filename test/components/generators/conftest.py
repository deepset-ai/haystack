# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat import chat_completion_chunk


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


class OpenAIMockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)
        self.mock_chunk = mock_chunk

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        yield self.mock_chunk


@pytest.fixture
def openai_mock_stream():
    """
    Fixture that returns a function to create MockStream instances with custom chunks
    """
    return OpenAIMockStream


@pytest.fixture
def openai_mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                {
                    "finish_reason": "stop",
                    "logprobs": None,
                    "index": 0,
                    "message": {"content": "Hello world!", "role": "assistant"},
                }
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def openai_mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = OpenAIMockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create
