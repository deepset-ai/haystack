# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY

import pytest
from openai.types import Reasoning, ResponseFormatText
from openai.types.responses import (
    FunctionTool,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from haystack.components.generators.chat.openai_responses import (
    _convert_chat_message_to_responses_api_format,
    _convert_response_chunk_to_streaming_chunk,
    _convert_streaming_chunks_to_chat_message,
)
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ReasoningContent,
    StreamingChunk,
    TextContent,
    ToolCall,
    ToolCallDelta,
    ToolCallResult,
)


@pytest.fixture
def openai_responses_streaming_chunks_with_tool_call():
    return [
        StreamingChunk(
            content="",
            meta={
                "received_at": ANY,
                "response": {
                    "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    "created_at": 1761907188.0,
                    "model": "gpt-5-mini-2025-08-07",
                    "object": "response",
                    "output": [],
                    "tools": [
                        {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                            "strict": False,
                            "type": "function",
                            "description": "useful to determine the weather in a given location",
                        }
                    ],
                    "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                    "usage": None,
                },
                "sequence_number": 0,
                "type": "response.created",
            },
        ),
        StreamingChunk(
            content="",
            meta={"received_at": ANY},
            index=0,
            start=True,
            reasoning=ReasoningContent(
                reasoning_text="",
                extra={
                    "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                    "summary": [],
                    "type": "reasoning",
                },
            ),
        ),
        StreamingChunk(
            content="",
            meta={
                "item": {
                    "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                    "summary": [],
                    "type": "reasoning",
                },
                "output_index": 0,
                "sequence_number": 3,
                "type": "response.output_item.done",
                "received_at": ANY,
            },
            index=0,
        ),
        StreamingChunk(
            content="",
            meta={"received_at": ANY},
            index=1,
            tool_calls=[
                ToolCallDelta(
                    index=1,
                    tool_name="weather",
                    arguments=None,
                    id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    extra={
                        "arguments": "",
                        "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                        "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        "name": "weather",
                        "status": "in_progress",
                        "type": "function_call",
                    },
                )
            ],
            start=True,
        ),
        StreamingChunk(
            content="",
            meta={"received_at": ANY},
            index=1,
            tool_calls=[
                ToolCallDelta(
                    index=1,
                    tool_name=None,
                    arguments='{"city":"Paris"}',
                    id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    extra={
                        "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        "output_index": 1,
                        "sequence_number": 5,
                        "type": "response.function_call_arguments.delta",
                        "obfuscation": "PySUcQ59ZZRkOm",
                    },
                )
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "received_at": ANY,
                "arguments": '{"city":"Paris"}',
                "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                "name": "weather",
                "output_index": 1,
                "sequence_number": 10,
                "type": "response.function_call_arguments.done",
            },
            index=1,
        ),
        StreamingChunk(
            content="",
            meta={
                "received_at": ANY,
                "response": {
                    "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    "created_at": 1761907188.0,
                    "metadata": {},
                    "model": "gpt-5-mini-2025-08-07",
                    "object": "response",
                    "output": [
                        {
                            "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                            "summary": [],
                            "type": "reasoning",
                        },
                        {
                            "arguments": '{"city":"Paris"}',
                            "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                            "name": "weather",
                            "type": "function_call",
                            "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "status": "completed",
                        },
                    ],
                    "tools": [
                        {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                            "strict": False,
                            "type": "function",
                            "description": "useful to determine the weather in a given location",
                        }
                    ],
                    "top_p": 1.0,
                    "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                    "usage": {
                        "input_tokens": 62,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 83,
                        "output_tokens_details": {"reasoning_tokens": 64},
                        "total_tokens": 145,
                    },
                    "store": True,
                },
                "sequence_number": 12,
                "type": "response.completed",
            },
            finish_reason="tool_calls",
        ),
    ]


class TestConversionToStreamingChunks:
    def test_convert_streaming_chunks_to_chat_message_with_tool_call_empty_reasoning(
        self, openai_responses_streaming_chunks_with_tool_call
    ):
        chat_message = _convert_streaming_chunks_to_chat_message(openai_responses_streaming_chunks_with_tool_call)
        assert chat_message == ChatMessage(
            _role="assistant",
            _content=[
                ReasoningContent(
                    reasoning_text="",
                    extra={"id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c", "type": "reasoning"},
                ),
                ToolCall(
                    tool_name="weather",
                    arguments={"city": "Paris"},
                    id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    extra={"call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv"},
                ),
            ],
            _name=None,
            _meta={
                "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                "created_at": 1761907188.0,
                "metadata": {},
                "model": "gpt-5-mini-2025-08-07",
                "object": "response",
                "tools": [
                    {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                        "type": "function",
                        "description": "useful to determine the weather in a given location",
                    }
                ],
                "top_p": 1.0,
                "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                "usage": {
                    "input_tokens": 62,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 83,
                    "output_tokens_details": {"reasoning_tokens": 64},
                    "total_tokens": 145,
                },
                "store": True,
            },
        )

    def test_convert_only_text(self):
        openai_chunks = [
            ResponseCreatedEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    service_tier="auto",
                    status="in_progress",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=0,
                type="response.created",
            ),
            ResponseInProgressEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    service_tier="auto",
                    status="in_progress",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=1,
                type="response.in_progress",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseReasoningItem(
                    id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=2,
                type="response.output_item.added",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=3,
                type="response.output_item.done",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseOutputMessage(
                    id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    content=[],
                    role="assistant",
                    status="in_progress",
                    type="message",
                ),
                output_index=1,
                sequence_number=4,
                type="response.output_item.added",
            ),
            ResponseContentPartAddedEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                output_index=1,
                part=ResponseOutputText(annotations=[], text="", type="output_text", logprobs=[]),
                sequence_number=5,
                type="response.content_part.added",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="Germany",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=6,
                type="response.output_text.delta",
                obfuscation="EV5gCoyiD",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=":",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=7,
                type="response.output_text.delta",
                obfuscation="EkdNXp1EE2Cgj8z",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=" Berlin",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=8,
                type="response.output_text.delta",
                obfuscation="1eS0q9aye",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="\n",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=9,
                type="response.output_text.delta",
                obfuscation="H9Ict3F41DwGS4a",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta="France",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=10,
                type="response.output_text.delta",
                obfuscation="4vxrblWURx",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=":",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=11,
                type="response.output_text.delta",
                obfuscation="B1CMJsNGhhqIz5K",
            ),
            ResponseTextDeltaEvent(
                content_index=0,
                delta=" Paris",
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=12,
                type="response.output_text.delta",
                obfuscation="ojbz89bS7j",
            ),
            ResponseTextDoneEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                logprobs=[],
                output_index=1,
                sequence_number=13,
                text="Germany: Berlin\nFrance: Paris",
                type="response.output_text.done",
            ),
            ResponseContentPartDoneEvent(
                content_index=0,
                item_id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                output_index=1,
                part=ResponseOutputText(
                    annotations=[], text="Germany: Berlin\nFrance: Paris", type="output_text", logprobs=[]
                ),
                sequence_number=14,
                type="response.content_part.done",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseOutputMessage(
                    id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="Germany: Berlin\nFrance: Paris", type="output_text", logprobs=[]
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=1,
                sequence_number=15,
                type="response.output_item.done",
            ),
            ResponseCompletedEvent(
                response=Response(
                    id="resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                    created_at=1762418678.0,
                    error=None,
                    incomplete_details=None,
                    instructions=None,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[
                        ResponseReasoningItem(
                            id="rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b", summary=[], type="reasoning"
                        ),
                        ResponseOutputMessage(
                            id="msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                            content=[
                                ResponseOutputText(
                                    annotations=[],
                                    text="Germany: Berlin\nFrance: Paris",
                                    type="output_text",
                                    logprobs=[],
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                        ),
                    ],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[],
                    top_p=1.0,
                    background=False,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    safety_identifier=None,
                    service_tier="default",
                    status="completed",
                    text=ResponseTextConfig(format=ResponseFormatText(type="text"), verbosity="medium"),
                    top_logprobs=0,
                    truncation="disabled",
                    usage=ResponseUsage(
                        input_tokens=15,
                        input_tokens_details=InputTokensDetails(cached_tokens=0),
                        output_tokens=77,
                        output_tokens_details=OutputTokensDetails(reasoning_tokens=64),
                        total_tokens=92,
                    ),
                    prompt_cache_retention=None,
                    store=True,
                ),
                sequence_number=16,
                type="response.completed",
            ),
        ]
        streaming_chunks = []
        for chunk in openai_chunks:
            streaming_chunk = _convert_response_chunk_to_streaming_chunk(chunk, previous_chunks=streaming_chunks)
            streaming_chunks.append(streaming_chunk)

        assert streaming_chunks == [
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "service_tier": "auto",
                        "status": "in_progress",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 0,
                    "type": "response.created",
                },
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "service_tier": "auto",
                        "status": "in_progress",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 1,
                    "type": "response.in_progress",
                },
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=0,
                start=True,
                reasoning=ReasoningContent(
                    reasoning_text="",
                    extra={
                        "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                        "summary": [],
                        "type": "reasoning",
                    },
                ),
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                        "summary": [],
                        "type": "reasoning",
                    },
                    "output_index": 0,
                    "sequence_number": 3,
                    "type": "response.output_item.done",
                },
                index=0,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                        "content": [],
                        "role": "assistant",
                        "status": "in_progress",
                        "type": "message",
                    },
                    "output_index": 1,
                    "sequence_number": 4,
                    "type": "response.output_item.added",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "output_index": 1,
                    "part": {"annotations": [], "text": "", "type": "output_text", "logprobs": []},
                    "sequence_number": 5,
                    "type": "response.content_part.added",
                },
                index=1,
            ),
            StreamingChunk(
                content="Germany",
                meta={
                    "content_index": 0,
                    "delta": "Germany",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 6,
                    "type": "response.output_text.delta",
                    "obfuscation": "EV5gCoyiD",
                    "received_at": ANY,
                },
                index=1,
                start=True,
            ),
            StreamingChunk(
                content=":",
                meta={
                    "content_index": 0,
                    "delta": ":",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 7,
                    "type": "response.output_text.delta",
                    "obfuscation": "EkdNXp1EE2Cgj8z",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=" Berlin",
                meta={
                    "content_index": 0,
                    "delta": " Berlin",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 8,
                    "type": "response.output_text.delta",
                    "obfuscation": "1eS0q9aye",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="\n",
                meta={
                    "content_index": 0,
                    "delta": "\n",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 9,
                    "type": "response.output_text.delta",
                    "obfuscation": "H9Ict3F41DwGS4a",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="France",
                meta={
                    "content_index": 0,
                    "delta": "France",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 10,
                    "type": "response.output_text.delta",
                    "obfuscation": "4vxrblWURx",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=":",
                meta={
                    "content_index": 0,
                    "delta": ":",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 11,
                    "type": "response.output_text.delta",
                    "obfuscation": "B1CMJsNGhhqIz5K",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content=" Paris",
                meta={
                    "content_index": 0,
                    "delta": " Paris",
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 12,
                    "type": "response.output_text.delta",
                    "obfuscation": "ojbz89bS7j",
                    "received_at": ANY,
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "logprobs": [],
                    "output_index": 1,
                    "sequence_number": 13,
                    "text": "Germany: Berlin\nFrance: Paris",
                    "type": "response.output_text.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "content_index": 0,
                    "item_id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                    "output_index": 1,
                    "part": {
                        "annotations": [],
                        "text": "Germany: Berlin\nFrance: Paris",
                        "type": "output_text",
                        "logprobs": [],
                    },
                    "sequence_number": 14,
                    "type": "response.content_part.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "item": {
                        "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                        "content": [
                            {
                                "annotations": [],
                                "text": "Germany: Berlin\nFrance: Paris",
                                "type": "output_text",
                                "logprobs": [],
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    },
                    "output_index": 1,
                    "sequence_number": 15,
                    "type": "response.output_item.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_0a8811e62a95217b00690c5ff62c14819596eae387d116f285",
                        "created_at": 1762418678.0,
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [
                            {
                                "id": "rs_0a8811e62a95217b00690c5ff70a308195a8207d7eb43f1d5b",
                                "summary": [],
                                "type": "reasoning",
                            },
                            {
                                "id": "msg_0a8811e62a95217b00690c5ff88f6c8195b037e57d327a1ee0",
                                "content": [
                                    {
                                        "annotations": [],
                                        "text": "Germany: Berlin\nFrance: Paris",
                                        "type": "output_text",
                                        "logprobs": [],
                                    }
                                ],
                                "role": "assistant",
                                "status": "completed",
                                "type": "message",
                            },
                        ],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "safety_identifier": None,
                        "service_tier": "default",
                        "status": "completed",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 15,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 77,
                            "output_tokens_details": {"reasoning_tokens": 64},
                            "total_tokens": 92,
                        },
                        "prompt_cache_retention": None,
                        "store": True,
                    },
                    "sequence_number": 16,
                    "type": "response.completed",
                },
                finish_reason="stop",
            ),
        ]

    def test_convert_only_function_call(self):
        chunks = [
            ResponseCreatedEvent(
                response=Response(
                    id="resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    created_at=1761907188.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[
                        FunctionTool(
                            name="weather",
                            parameters={
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                            strict=False,
                            type="function",
                            description="useful to determine the weather in a given location",
                        )
                    ],
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    usage=None,
                ),
                sequence_number=0,
                type="response.created",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseReasoningItem(
                    id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=2,
                type="response.output_item.added",
            ),
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=3,
                type="response.output_item.done",
            ),
            ResponseOutputItemAddedEvent(
                item=ResponseFunctionToolCall(
                    arguments="",
                    call_id="call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                    name="weather",
                    type="function_call",
                    id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    status="in_progress",
                ),
                output_index=1,
                sequence_number=4,
                type="response.output_item.added",
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                delta='{"city":',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                output_index=1,
                sequence_number=5,
                type="response.function_call_arguments.delta",
                obfuscation="PySUcQ59ZZRkOm",
            ),
            ResponseFunctionCallArgumentsDeltaEvent(
                delta='"Paris"}',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                output_index=1,
                sequence_number=8,
                type="response.function_call_arguments.delta",
                obfuscation="INeMDAi1uAj",
            ),
            ResponseFunctionCallArgumentsDoneEvent(
                arguments='{"city":"Paris"}',
                item_id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                name="weather",  # added name here because pydantic complains otherwise API returns a none here
                output_index=1,
                sequence_number=10,
                type="response.function_call_arguments.done",
            ),
            ResponseCompletedEvent(
                response=Response(
                    id="resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                    created_at=1761907188.0,
                    metadata={},
                    model="gpt-5-mini-2025-08-07",
                    object="response",
                    output=[
                        ResponseReasoningItem(
                            id="rs_095b57053855eac100690491f54e308196878239be3ba6133c", summary=[], type="reasoning"
                        ),
                        ResponseFunctionToolCall(
                            arguments='{"city":"Paris"}',
                            call_id="call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                            name="weather",
                            type="function_call",
                            id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            status="completed",
                        ),
                    ],
                    parallel_tool_calls=True,
                    temperature=1.0,
                    tool_choice="auto",
                    tools=[
                        FunctionTool(
                            name="weather",
                            parameters={
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                            strict=False,
                            type="function",
                            description="useful to determine the weather in a given location",
                        )
                    ],
                    top_p=1.0,
                    reasoning=Reasoning(effort="medium", generate_summary=None, summary=None),
                    usage=ResponseUsage(
                        input_tokens=62,
                        input_tokens_details=InputTokensDetails(cached_tokens=0),
                        output_tokens=83,
                        output_tokens_details=OutputTokensDetails(reasoning_tokens=64),
                        total_tokens=145,
                    ),
                    store=True,
                ),
                sequence_number=12,
                type="response.completed",
            ),
        ]

        streaming_chunks = []
        for chunk in chunks:
            streaming_chunk = _convert_response_chunk_to_streaming_chunk(chunk, previous_chunks=streaming_chunks)
            streaming_chunks.append(streaming_chunk)

        assert streaming_chunks == [
            # TODO Unneeded streaming chunk
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                        "created_at": 1761907188.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [
                            {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                                "strict": False,
                                "type": "function",
                                "description": "useful to determine the weather in a given location",
                            }
                        ],
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "usage": None,
                    },
                    "sequence_number": 0,
                    "type": "response.created",
                },
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=0,
                start=True,
                reasoning=ReasoningContent(
                    reasoning_text="",
                    extra={
                        "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                        "summary": [],
                        "type": "reasoning",
                    },
                ),
            ),
            StreamingChunk(
                content="",
                meta={
                    "item": {
                        "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                        "summary": [],
                        "type": "reasoning",
                    },
                    "output_index": 0,
                    "sequence_number": 3,
                    "type": "response.output_item.done",
                    "received_at": ANY,
                },
                index=0,
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name="weather",
                        arguments=None,
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "arguments": "",
                            "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                            "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "name": "weather",
                            "status": "in_progress",
                            "type": "function_call",
                        },
                    )
                ],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name=None,
                        arguments='{"city":',
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "output_index": 1,
                            "sequence_number": 5,
                            "type": "response.function_call_arguments.delta",
                            "obfuscation": "PySUcQ59ZZRkOm",
                        },
                    )
                ],
            ),
            StreamingChunk(
                content="",
                meta={"received_at": ANY},
                index=1,
                tool_calls=[
                    ToolCallDelta(
                        index=1,
                        tool_name=None,
                        arguments='"Paris"}',
                        id="fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                        extra={
                            "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                            "output_index": 1,
                            "sequence_number": 8,
                            "type": "response.function_call_arguments.delta",
                            "obfuscation": "INeMDAi1uAj",
                        },
                    )
                ],
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "arguments": '{"city":"Paris"}',
                    "item_id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                    "name": "weather",
                    "output_index": 1,
                    "sequence_number": 10,
                    "type": "response.function_call_arguments.done",
                },
                index=1,
            ),
            StreamingChunk(
                content="",
                meta={
                    "received_at": ANY,
                    "response": {
                        "id": "resp_095b57053855eac100690491f4e22c8196ac124365e8c70424",
                        "created_at": 1761907188.0,
                        "metadata": {},
                        "model": "gpt-5-mini-2025-08-07",
                        "object": "response",
                        "output": [
                            {
                                "id": "rs_095b57053855eac100690491f54e308196878239be3ba6133c",
                                "summary": [],
                                "type": "reasoning",
                            },
                            {
                                "arguments": '{"city":"Paris"}',
                                "call_id": "call_OZZXFm7SLb4F3Xg8a9XVVCvv",
                                "name": "weather",
                                "type": "function_call",
                                "id": "fc_095b57053855eac100690491f6a224819680e2f9c7cbc5a531",
                                "status": "completed",
                            },
                        ],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [
                            {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                                "strict": False,
                                "type": "function",
                                "description": "useful to determine the weather in a given location",
                            }
                        ],
                        "top_p": 1.0,
                        "reasoning": {"effort": "medium", "generate_summary": None, "summary": None},
                        "usage": {
                            "input_tokens": 62,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 83,
                            "output_tokens_details": {"reasoning_tokens": 64},
                            "total_tokens": 145,
                        },
                        "store": True,
                    },
                    "sequence_number": 12,
                    "type": "response.completed",
                },
                finish_reason="tool_calls",
            ),
        ]


class TestResponseToChatMessage:
    def test_convert_chat_message_to_responses_api_format(self):
        chat_message = ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=[
                TextContent(text="I need to use the functions.weather tool."),
                ReasoningContent(
                    reasoning_text="I need to use the functions.weather tool.",
                    extra={"id": "rs_0d13efdd", "type": "reasoning"},
                ),
                ToolCall(
                    tool_name="weather",
                    arguments={"location": "Berlin"},
                    id="fc_0d13efdd",
                    extra={"call_id": "call_a82vwFAIzku9SmBuQuecQSRq"},
                ),
            ],
            _name=None,
            # some keys are removed to keep the test concise
            _meta={
                "id": "resp_0d13efdd97aa4",
                "created_at": 1761148307.0,
                "model": "gpt-5-mini-2025-08-07",
                "object": "response",
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [
                    {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "additionalProperties": False,
                        },
                        "strict": False,
                        "type": "function",
                        "description": "A tool to get the weather",
                    }
                ],
                "top_p": 1.0,
                "reasoning": {"effort": "low", "summary": "detailed"},
                "usage": {"input_tokens": 59, "output_tokens": 19, "total_tokens": 78},
                "store": True,
            },
        )
        responses_api_format = _convert_chat_message_to_responses_api_format(chat_message)
        assert responses_api_format == [
            {
                "id": "rs_0d13efdd",
                "type": "reasoning",
                "summary": [{"text": "I need to use the functions.weather tool.", "type": "summary_text"}],
            },
            {
                "type": "function_call",
                "name": "weather",
                "arguments": '{"location": "Berlin"}',
                "id": "fc_0d13efdd",
                "call_id": "call_a82vwFAIzku9SmBuQuecQSRq",
            },
            {"content": "I need to use the functions.weather tool.", "role": "assistant"},
        ]
        # ToolCallResult cannot appear with other content
        tool_call_result = ChatMessage(
            _role=ChatRole.TOOL,
            _content=[
                ToolCallResult(
                    result="result",
                    origin=ToolCall(
                        id="fc_0d13efdd",
                        tool_name="weather",
                        arguments={"location": "Berlin"},
                        extra={"call_id": "call_a82vwFAIzku9SmBuQuecQSRq"},
                    ),
                    error=False,
                )
            ],
        )

        assert _convert_chat_message_to_responses_api_format(tool_call_result) == [
            {"call_id": "call_a82vwFAIzku9SmBuQuecQSRq", "output": "result", "type": "function_call_output"}
        ]
