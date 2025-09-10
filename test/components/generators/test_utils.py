# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import call, patch

from openai.types.chat import chat_completion_chunk

from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, print_streaming_chunk
from haystack.dataclasses import ComponentInfo, StreamingChunk, ToolCall, ToolCallDelta, ToolCallResult


def test_convert_streaming_chunks_to_chat_message_tool_calls_in_any_chunk():
    chunks = [
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.910076",
            },
            component_info=ComponentInfo(name="test", type="test"),
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id="call_ZOj5l67zhZOx6jqjg7ATQwb6",
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                            arguments="", name="rag_pipeline_tool"
                        ),
                        type="function",
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.913919",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            start=True,
            tool_calls=[
                ToolCallDelta(id="call_ZOj5l67zhZOx6jqjg7ATQwb6", tool_name="rag_pipeline_tool", arguments="", tool_call_index=0)
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"qu')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.914439",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments='{"qu', tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ery":')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924146",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments='ery":', tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Wher')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924420",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments=' "Wher', tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="e do")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944398",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments="e do", tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="es Ma")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944958",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments="es Ma", tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="rk liv")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.945507",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments="rk liv", tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='e?"}')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946018",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments='e?"}', tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1,
                        id="call_STxsYY69wVOvxWqopAt3uWTB",
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="", name="get_weather"),
                        type="function",
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946578",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=1,
            start=True,
            tool_calls=[
                ToolCallDelta(id="call_STxsYY69wVOvxWqopAt3uWTB", tool_name="get_weather", arguments="", tool_call_index=1)
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"ci')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946981",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=1,
            tool_calls=[ToolCallDelta(arguments='{"ci', tool_call_index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ty": ')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947411",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=1,
            tool_calls=[ToolCallDelta(arguments='ty": ', tool_call_index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='"Berli')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947643",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=1,
            tool_calls=[ToolCallDelta(arguments='"Berli', tool_call_index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='n"}')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947939",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=1,
            tool_calls=[ToolCallDelta(arguments='n"}', tool_call_index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": "tool_calls",
                "received_at": "2025-02-19T16:02:55.948772",
            },
            component_info=ComponentInfo(name="test", type="test"),
            finish_reason="tool_calls",
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.948772",
                "usage": {
                    "completion_tokens": 42,
                    "prompt_tokens": 282,
                    "total_tokens": 324,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                },
            },
            component_info=ComponentInfo(name="test", type="test"),
        ),
    ]

    # Convert chunks to a chat message
    result = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    assert not result.texts
    assert not result.text

    # Verify both tool calls were found and processed
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].id == "call_ZOj5l67zhZOx6jqjg7ATQwb6"
    assert result.tool_calls[0].tool_name == "rag_pipeline_tool"
    assert result.tool_calls[0].arguments == {"query": "Where does Mark live?"}
    assert result.tool_calls[1].id == "call_STxsYY69wVOvxWqopAt3uWTB"
    assert result.tool_calls[1].tool_name == "get_weather"
    assert result.tool_calls[1].arguments == {"city": "Berlin"}

    # Verify meta information
    assert result.meta["model"] == "gpt-4o-mini-2024-07-18"
    assert result.meta["finish_reason"] == "tool_calls"
    assert result.meta["chunk_index"] == 0
    assert result.meta["completion_start_time"] == "2025-02-19T16:02:55.910076"
    assert result.meta["usage"] == {
        "completion_tokens": 42,
        "prompt_tokens": 282,
        "total_tokens": 324,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
    }


def test_convert_streaming_chunk_to_chat_message_two_tool_calls_in_same_chunk():
    chunks = [
        StreamingChunk(
            content="",
            meta={
                "model": "mistral-small-latest",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "usage": None,
            },
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
                name=None,
            ),
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "mistral-small-latest",
                "chunk_index": 0,
                "finish_reason": "tool_calls",
                "usage": {
                    "completion_tokens": 35,
                    "prompt_tokens": 77,
                    "total_tokens": 112,
                    "completion_tokens_details": None,
                    "prompt_tokens_details": None,
                },
            },
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
                name=None,
            ),
            chunk_index=0,
            tool_calls=[
                ToolCallDelta(tool_call_index=0, tool_name="weather", arguments='{"city": "Paris"}', id="FL1FFlqUG"),
                ToolCallDelta(tool_call_index=1, tool_name="weather", arguments='{"city": "Berlin"}', id="xSuhp66iB"),
            ],
            start=True,
            finish_reason="tool_calls",
        ),
    ]

    # Convert chunks to a chat message
    result = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    assert not result.texts
    assert not result.text

    # Verify both tool calls were found and processed
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].id == "FL1FFlqUG"
    assert result.tool_calls[0].tool_name == "weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}
    assert result.tool_calls[1].id == "xSuhp66iB"
    assert result.tool_calls[1].tool_name == "weather"
    assert result.tool_calls[1].arguments == {"city": "Berlin"}


def test_convert_streaming_chunk_to_chat_message_empty_tool_call_delta():
    chunks = [
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.910076",
            },
            component_info=ComponentInfo(name="test", type="test"),
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id="call_ZOj5l67zhZOx6jqjg7ATQwb6",
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                            arguments='{"query":', name="rag_pipeline_tool"
                        ),
                        type="function",
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.913919",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            start=True,
            tool_calls=[
                ToolCallDelta(
                    id="call_ZOj5l67zhZOx6jqjg7ATQwb6", tool_name="rag_pipeline_tool", arguments='{"query":', tool_call_index=0
                )
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                            arguments=' "Where does Mark live?"}'
                        ),
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924420",
            },
            component_info=ComponentInfo(name="test", type="test"),
            chunk_index=0,
            tool_calls=[ToolCallDelta(arguments=' "Where does Mark live?"}', tool_call_index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction()
                    )
                ],
                "finish_reason": "tool_calls",
                "received_at": "2025-02-19T16:02:55.948772",
            },
            tool_calls=[ToolCallDelta(tool_call_index=0)],
            component_info=ComponentInfo(name="test", type="test"),
            finish_reason="tool_calls",
            chunk_index=0,
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "chunk_index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.948772",
                "usage": {
                    "completion_tokens": 42,
                    "prompt_tokens": 282,
                    "total_tokens": 324,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                },
            },
            component_info=ComponentInfo(name="test", type="test"),
        ),
    ]

    # Convert chunks to a chat message
    result = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    assert not result.texts
    assert not result.text

    # Verify both tool calls were found and processed
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_ZOj5l67zhZOx6jqjg7ATQwb6"
    assert result.tool_calls[0].tool_name == "rag_pipeline_tool"
    assert result.tool_calls[0].arguments == {"query": "Where does Mark live?"}
    assert result.meta["finish_reason"] == "tool_calls"


def test_convert_streaming_chunk_to_chat_message_with_empty_tool_call_arguments():
    chunks = [
        # Message start with input tokens
        StreamingChunk(content="",
            meta={
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-20250514",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 25, "output_tokens": 0},
                },
            },chunk_index=0,
            tool_calls=[],
            tool_call_result=None,
            start=True,
            finish_reason=None,
        ),
        # Initial text content
        StreamingChunk(content="",
            meta={"type": "content_block_start", "chunk_index": 0, "content_block": {"type": "text", "text": ""}},chunk_index=1,
            tool_calls=[],
            tool_call_result=None,
            start=True,
            finish_reason=None,
        ),
        StreamingChunk(content="Let me check",
            meta={"type": "content_block_delta", "chunk_index": 0, "delta": {"type": "text_delta", "text": "Let me check"}},chunk_index=2,
            tool_calls=[],
            tool_call_result=None,
            start=False,
            finish_reason=None,
        ),
        StreamingChunk(content=" the weather",
            meta={"type": "content_block_delta", "chunk_index": 0, "delta": {"type": "text_delta", "text": " the weather"}},chunk_index=3,
            tool_calls=[],
            tool_call_result=None,
            start=False,
            finish_reason=None,
        ),
        # Tool use content
        StreamingChunk(content="",
            meta={
                "type": "content_block_start",
                "chunk_index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {}},
            },chunk_index=5,
            tool_calls=[ToolCallDelta(tool_call_index=1, id="toolu_123", tool_name="weather", arguments=None)],
            tool_call_result=None,
            start=True,
            finish_reason=None,
        ),
        StreamingChunk(content="",
            meta={"type": "content_block_delta", "chunk_index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},chunk_index=7,
            tool_calls=[ToolCallDelta(tool_call_index=1, id=None, tool_name=None, arguments="")],
            tool_call_result=None,
            start=False,
            finish_reason=None,
        ),
        # Final message delta
        StreamingChunk(content="",
            meta={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"completion_tokens": 40},
            },chunk_index=8,
            tool_calls=[],
            tool_call_result=None,
            start=False,
            finish_reason="tool_calls",
        ),
    ]

    message = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    assert message.texts == ["Let me check the weather"]
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].arguments == {}
    assert message.tool_calls[0].id == "toolu_123"
    assert message.tool_calls[0].tool_name == "weather"


def test_print_streaming_chunk_content_only():
    chunk = StreamingChunk(
        content="Hello, world!",
        meta={"model": "test-model"},
        component_info=ComponentInfo(name="test", type="test"),
        start=True,
    )
    with patch("builtins.print") as mock_print:
        print_streaming_chunk(chunk)
        expected_calls = [call("[ASSISTANT]\n", flush=True, end=""), call("Hello, world!", flush=True, end="")]
        mock_print.assert_has_calls(expected_calls)


def test_print_streaming_chunk_tool_call():
    chunk = StreamingChunk(
        content="",
        meta={"model": "test-model"},
        component_info=ComponentInfo(name="test", type="test"),
        start=True,
        chunk_index=0,
        tool_calls=[ToolCallDelta(id="call_123", tool_name="test_tool", arguments='{"param": "value"}', tool_call_index=0)],
    )
    with patch("builtins.print") as mock_print:
        print_streaming_chunk(chunk)
        expected_calls = [
            call("[TOOL CALL]\nTool: test_tool \nArguments: ", flush=True, end=""),
            call('{"param": "value"}', flush=True, end=""),
        ]
        mock_print.assert_has_calls(expected_calls)


def test_print_streaming_chunk_tool_call_result():
    chunk = StreamingChunk(
        content="",
        meta={"model": "test-model"},
        component_info=ComponentInfo(name="test", type="test"),
        chunk_index=0,
        tool_call_result=ToolCallResult(
            result="Tool execution completed successfully",
            origin=ToolCall(id="call_123", tool_name="test_tool", arguments={}),
            error=False,
        ),
    )
    with patch("builtins.print") as mock_print:
        print_streaming_chunk(chunk)
        expected_calls = [call("[TOOL RESULT]\nTool execution completed successfully", flush=True, end="")]
        mock_print.assert_has_calls(expected_calls)


def test_print_streaming_chunk_with_finish_reason():
    chunk = StreamingChunk(
        content="Final content.",
        meta={"model": "test-model"},
        component_info=ComponentInfo(name="test", type="test"),
        start=True,
        finish_reason="stop",
    )
    with patch("builtins.print") as mock_print:
        print_streaming_chunk(chunk)
        expected_calls = [
            call("[ASSISTANT]\n", flush=True, end=""),
            call("Final content.", flush=True, end=""),
            call("\n\n", flush=True, end=""),
        ]
        mock_print.assert_has_calls(expected_calls)


def test_print_streaming_chunk_empty_chunk():
    chunk = StreamingChunk(
        content="", meta={"model": "test-model"}, component_info=ComponentInfo(name="test", type="test")
    )
    with patch("builtins.print") as mock_print:
        print_streaming_chunk(chunk)
        mock_print.assert_not_called()
