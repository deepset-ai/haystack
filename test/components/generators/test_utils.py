# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from openai.types.chat import chat_completion_chunk

from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import ComponentInfo, StreamingChunk, ToolCallDelta


def test_convert_streaming_chunks_to_chat_message_tool_calls_in_any_chunk():
    chunks = [
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
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
                "index": 0,
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
            index=0,
            start=True,
            tool_calls=[
                ToolCallDelta(id="call_ZOj5l67zhZOx6jqjg7ATQwb6", tool_name="rag_pipeline_tool", arguments="", index=0)
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"qu')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.914439",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments='{"qu', index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ery":')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924146",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments='ery":', index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Wher')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924420",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments=' "Wher', index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="e do")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944398",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments="e do", index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="es Ma")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944958",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments="es Ma", index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="rk liv")
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.945507",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments="rk liv", index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='e?"}')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946018",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=0,
            tool_calls=[ToolCallDelta(arguments='e?"}', index=0)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
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
            index=1,
            start=True,
            tool_calls=[
                ToolCallDelta(id="call_STxsYY69wVOvxWqopAt3uWTB", tool_name="get_weather", arguments="", index=1)
            ],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"ci')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946981",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=1,
            tool_calls=[ToolCallDelta(arguments='{"ci', index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ty": ')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947411",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=1,
            tool_calls=[ToolCallDelta(arguments='ty": ', index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='"Berli')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947643",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=1,
            tool_calls=[ToolCallDelta(arguments='"Berli', index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1, function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='n"}')
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947939",
            },
            component_info=ComponentInfo(name="test", type="test"),
            index=1,
            tool_calls=[ToolCallDelta(arguments='n"}', index=1)],
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
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
                "index": 0,
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
    assert result.meta["index"] == 0
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
                "index": 0,
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
                "index": 0,
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
            index=0,
            tool_calls=[
                ToolCallDelta(index=0, tool_name="weather", arguments='{"city": "Paris"}', id="FL1FFlqUG"),
                ToolCallDelta(index=1, tool_name="weather", arguments='{"city": "Berlin"}', id="xSuhp66iB"),
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
