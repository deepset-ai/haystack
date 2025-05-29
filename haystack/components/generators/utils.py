# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from haystack import logging
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall

logger = logging.getLogger(__name__)


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Callback function to handle and display streaming output chunks.

    This function processes a `StreamingChunk` object by:
    - Printing tool call metadata (if any), including function names and arguments, as they arrive.
    - Printing tool call results when available.
    - Printing the main content (e.g., text tokens) of the chunk as it is received.

    The function outputs data directly to stdout and flushes output buffers to ensure immediate display during
    streaming.

    :param chunk: A chunk of streaming data containing content and optional metadata, such as tool calls and
        tool results.
    """
    # Print tool call metadata if available (from ChatGenerator)
    if tool_calls := chunk.meta.get("tool_calls"):
        for tool_call in tool_calls:
            # Convert to dict if tool_call is a ChoiceDeltaToolCall
            tool_call_dict: Dict[str, Any] = (
                tool_call.to_dict() if isinstance(tool_call, ChoiceDeltaToolCall) else tool_call
            )

            if function := tool_call_dict.get("function"):
                if name := function.get("name"):
                    print("\n\n[TOOL CALL]\n", flush=True, end="")
                    print(f"Tool: {name} ", flush=True, end="")
                    print("\nArguments: ", flush=True, end="")

                if arguments := function.get("arguments"):
                    print(arguments, flush=True, end="")

    # Print tool call results if available (from ToolInvoker)
    if tool_result := chunk.meta.get("tool_result"):
        print(f"\n\n[TOOL RESULT]\n{tool_result}", flush=True, end="")

    # Print the main content of the chunk (from ChatGenerator)
    if content := chunk.content:
        print(content, flush=True, end="")

    # End of LLM assistant message so we add two new lines
    # This ensures spacing between multiple LLM messages (e.g. Agent)
    if chunk.meta.get("finish_reason") is not None:
        print("\n\n", flush=True, end="")


def _convert_streaming_chunks_to_chat_message(chunks: List[StreamingChunk]) -> ChatMessage:
    """
    Connects the streaming chunks into a single ChatMessage.

    :param chunks: The list of all `StreamingChunk` objects.

    :returns: The ChatMessage.
    """
    text = "".join([chunk.content for chunk in chunks])
    tool_calls = []

    # Process tool calls if present in any chunk
    tool_call_data: Dict[str, Dict[str, str]] = {}  # Track tool calls by index
    for chunk_payload in chunks:
        tool_calls_meta = chunk_payload.meta.get("tool_calls")
        if tool_calls_meta is not None:
            for delta in tool_calls_meta:
                # We use the index of the tool call to track it across chunks since the ID is not always provided
                if delta.index not in tool_call_data:
                    tool_call_data[delta.index] = {"id": "", "name": "", "arguments": ""}

                # Save the ID if present
                if delta.id is not None:
                    tool_call_data[delta.index]["id"] = delta.id

                if delta.function is not None:
                    if delta.function.name is not None:
                        tool_call_data[delta.index]["name"] += delta.function.name
                    if delta.function.arguments is not None:
                        tool_call_data[delta.index]["arguments"] += delta.function.arguments

    # Convert accumulated tool call data into ToolCall objects
    for call_data in tool_call_data.values():
        try:
            arguments = json.loads(call_data["arguments"])
            tool_calls.append(ToolCall(id=call_data["id"], tool_name=call_data["name"], arguments=arguments))
        except json.JSONDecodeError:
            logger.warning(
                "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                _id=call_data["id"],
                _name=call_data["name"],
                _arguments=call_data["arguments"],
            )

    # finish_reason can appear in different places so we look for the last one
    finish_reasons = [
        chunk.meta.get("finish_reason") for chunk in chunks if chunk.meta.get("finish_reason") is not None
    ]
    finish_reason = finish_reasons[-1] if finish_reasons else None

    meta = {
        "model": chunks[-1].meta.get("model"),
        "index": 0,
        "finish_reason": finish_reason,
        "completion_start_time": chunks[0].meta.get("received_at"),  # first chunk received
        "usage": chunks[-1].meta.get("usage"),  # last chunk has the final usage data if available
    }

    return ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, meta=meta)
