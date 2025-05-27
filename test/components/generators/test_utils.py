# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from openai.types.chat import chat_completion_chunk

from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import ComponentInfo, StreamingChunk


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
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"qu', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.914439",
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
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ery":', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924146",
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
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments=' "Wher', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.924420",
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
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="e do", name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944398",
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
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="es Ma", name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.944958",
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
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments="rk liv", name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.945507",
            },
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='e?"}', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946018",
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
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "gpt-4o-mini-2024-07-18",
                "index": 0,
                "tool_calls": [
                    chat_completion_chunk.ChoiceDeltaToolCall(
                        index=1,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='{"ci', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.946981",
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
                        index=1,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='ty": ', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947411",
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
                        index=1,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='"Berli', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947643",
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
                        index=1,
                        id=None,
                        function=chat_completion_chunk.ChoiceDeltaToolCallFunction(arguments='n"}', name=None),
                        type=None,
                    )
                ],
                "finish_reason": None,
                "received_at": "2025-02-19T16:02:55.947939",
            },
            component_info=ComponentInfo(name="test", type="test"),
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
