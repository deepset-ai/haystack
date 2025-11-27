# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncStream, Stream
from openai.types import Reasoning
from openai.types.chat import ChatCompletion, ChatCompletionChunk, chat_completion_chunk
from openai.types.responses import (
    Response,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails


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


class OpenAIAsyncMockStream(AsyncStream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk):
        self.mock_chunk = mock_chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Only yield once, then stop iteration
        if not hasattr(self, "_done"):
            self._done = True
            return self.mock_chunk
        raise StopAsyncIteration


@pytest.fixture
def openai_mock_stream():
    """
    Fixture that returns a function to create MockStream instances with custom chunks
    """
    return OpenAIMockStream


@pytest.fixture
def openai_mock_stream_async():
    """
    Fixture that returns a function to create AsyncMockStream instances with custom chunks
    """
    return OpenAIAsyncMockStream


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
async def openai_mock_async_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for async tests
    """
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_chat_completion_create:
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
            usage=None,
        )
        mock_chat_completion_create.return_value = OpenAIMockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


@pytest.fixture
async def openai_mock_async_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for async tests
    """
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_chat_completion_create:
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
            usage=None,
        )
        mock_chat_completion_create.return_value = OpenAIAsyncMockStream(completion)
        yield mock_chat_completion_create


@pytest.fixture
def openai_mock_responses():
    """
    Mock a fully populated non-streaming Response returned by the
    OpenAI Responses API (client.responses.create).
    """

    with patch("openai.resources.responses.Responses.create") as mock_create:
        # Build the Response object exactly like the one you provided
        mock_response = Response(
            id="resp_mock_123",
            created_at=float(datetime.now().timestamp()),
            metadata={},
            model="gpt-5-mini-2025-08-07",
            object="response",
            output=[
                ResponseReasoningItem(
                    id="rs_mock_1",
                    type="reasoning",
                    summary=[
                        Summary(
                            text=(
                                "**Providing concise information**\n\n"
                                "The question is simple: the answer is Paris. "
                                "It’s useful to mention that Paris is the capital and a major "
                                "city in France. There’s really no need for extra details in this "
                                "case, so I’ll keep it concise and straightforward."
                            ),
                            type="summary_text",
                        )
                    ],
                ),
                ResponseOutputMessage(
                    id="msg_mock_1",
                    role="assistant",
                    type="message",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            text="The capital of France is Paris.", type="output_text", logprobs=None, annotations=[]
                        )
                    ],
                ),
            ],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            reasoning=Reasoning(effort="low", generate_summary=None, summary="auto"),
            usage=ResponseUsage(
                input_tokens=11,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=13,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=24,
            ),
            user=None,
            billing={"payer": "developer"},
            prompt_cache_retention=None,
            store=True,
        )

        mock_create.return_value = mock_response
        yield mock_create


@pytest.fixture
def openai_mock_async_responses():
    """
    Mock a fully populated non-streaming Response returned by the
    OpenAI Responses API (client.responses.create).
    """

    with patch("openai.resources.responses.AsyncResponses.create") as mock_create:
        # Build the Response object exactly like the one you provided
        mock_response = Response(
            id="resp_mock_123",
            created_at=float(datetime.now().timestamp()),
            metadata={},
            model="gpt-5-mini-2025-08-07",
            object="response",
            output=[
                ResponseReasoningItem(
                    id="rs_mock_1",
                    type="reasoning",
                    summary=[
                        Summary(
                            text=(
                                "**Providing concise information**\n\n"
                                "The question is simple: the answer is Paris. "
                                "It’s useful to mention that Paris is the capital and a major "
                                "city in France. There’s really no need for extra details in this "
                                "case, so I’ll keep it concise and straightforward."
                            ),
                            type="summary_text",
                        )
                    ],
                ),
                ResponseOutputMessage(
                    id="msg_mock_1",
                    role="assistant",
                    type="message",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            text="The capital of France is Paris.", type="output_text", annotations=[], logprobs=None
                        )
                    ],
                ),
            ],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            reasoning=Reasoning(effort="low", generate_summary=None, summary="auto"),
            usage=ResponseUsage(
                input_tokens=11,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=13,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=24,
            ),
            user=None,
            billing={"payer": "developer"},
            prompt_cache_retention=None,
            store=True,
        )

        mock_create.return_value = mock_response
        yield mock_create


@pytest.fixture
def openai_mock_responses_stream_text_delta():
    """
    Mock the Responses API streaming text-delta event (sync)
    and reuse it for tests.
    """

    with patch("openai.resources.responses.Responses.create") as mock_responses_create:
        event = ResponseTextDeltaEvent(
            # required fields in the current SDK
            content_index=0,
            delta="The capital of France is Paris.",
            item_id="item_1",
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )

        # Your OpenAIMockStream should iterate over this event
        mock_responses_create.return_value = OpenAIMockStream(event, cast_to=None, response=None, client=None)
        yield mock_responses_create


@pytest.fixture
async def openai_mock_async_responses_stream_text_delta():
    """
    Mock the Responses API streaming text-delta event (async)
    and reuse it for async tests.
    """

    with patch("openai.resources.responses.AsyncResponses.create", new_callable=AsyncMock) as mock_responses_create:
        event = ResponseTextDeltaEvent(
            content_index=0,
            delta="Hello",
            item_id="item_1",
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )

        mock_responses_create.return_value = OpenAIAsyncMockStream(event)
        yield mock_responses_create


@pytest.fixture
def openai_mock_responses_reasoning_summary_delta():
    """
    Mock a Responses API *streaming* reasoning summary text delta event (sync).
    """

    with patch("openai.resources.responses.Responses.create") as mock_responses_create:
        start_event = ResponseOutputItemAddedEvent(
            item=ResponseReasoningItem(
                id="rs_094e3f8beffcca02006928978067848190b477543eddbf32b3",
                summary=[],
                type="reasoning",
                content=None,
                encrypted_content=None,
                status=None,
            ),
            output_index=0,
            sequence_number=2,
            type="response.output_item.added",
        )

        event = ResponseReasoningSummaryTextDeltaEvent(
            delta="I need to check the capital of France.",
            item_id="rs_01e88f7d57f9a2f70069284d2170c48193918c04f85244cf7c",
            output_index=0,
            sequence_number=4,
            summary_index=0,
            type="response.reasoning_summary_text.delta",
            obfuscation="cGcv5W5F",
        )

        # Create a custom stream that yields both events sequentially
        class MultiEventMockStream(OpenAIMockStream):
            def __init__(self, *events, **kwargs):
                self.events = events
                super().__init__(events[0] if events else None, **kwargs)

            def __stream__(self):
                for event in self.events:
                    yield event

        mock_responses_create.return_value = MultiEventMockStream(
            start_event, event, cast_to=None, response=None, client=None
        )

        yield mock_responses_create
