# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, List

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
    if chunk.start and chunk.index and chunk.index > 0:
        # If this is the start of a new content block but not the first content block, print two new lines
        print("\n\n", flush=True, end="")

    ## Tool Call streaming
    if chunk.tool_calls:
        # Typically, if there are multiple tool calls in the chunk this means that the tool calls are fully formed and
        # not just a delta.
        for tool_call in chunk.tool_calls:
            # If chunk.start is True indicates beginning of a tool call
            # Also presence of tool_call.tool_name indicates the start of a tool call too
            if chunk.start:
                # If there is more than one tool call in the chunk, we print two new lines to separate them
                # We know there is more than one tool call if the index of the tool call is greater than the index of
                # the chunk.
                if chunk.index and tool_call.index > chunk.index:
                    print("\n\n", flush=True, end="")

                print(f"[TOOL CALL]\nTool: {tool_call.tool_name} \nArguments: ", flush=True, end="")

            # print the tool arguments
            if tool_call.arguments:
                print(tool_call.arguments, flush=True, end="")

    ## Tool Call Result streaming
    # Print tool call results if available (from ToolInvoker)
    if chunk.tool_call_result:
        # Tool Call Result is fully formed so delta accumulation is not needed
        print(f"[TOOL RESULT]\n{chunk.tool_call_result.result}", flush=True, end="")

    ## Normal content streaming
    # Print the main content of the chunk (from ChatGenerator)
    if chunk.content:
        if chunk.start:
            print("[ASSISTANT]\n", flush=True, end="")
        print(chunk.content, flush=True, end="")

    # End of LLM assistant message so we add two new lines
    # This ensures spacing between multiple LLM messages (e.g. Agent) or multiple Tool Call Results
    if chunk.finish_reason is not None:
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
    tool_call_data: Dict[int, Dict[str, str]] = {}  # Track tool calls by index
    for chunk in chunks:
        if chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                # We use the index of the tool_call to track the tool call across chunks since the ID is not always
                # provided
                if tool_call.index not in tool_call_data:
                    tool_call_data[tool_call.index] = {"id": "", "name": "", "arguments": ""}

                # Save the ID if present
                if tool_call.id is not None:
                    tool_call_data[tool_call.index]["id"] = tool_call.id

                if tool_call.tool_name is not None:
                    tool_call_data[tool_call.index]["name"] += tool_call.tool_name
                if tool_call.arguments is not None:
                    tool_call_data[tool_call.index]["arguments"] += tool_call.arguments

    # Convert accumulated tool call data into ToolCall objects
    sorted_keys = sorted(tool_call_data.keys())
    for key in sorted_keys:
        tool_call_dict = tool_call_data[key]
        try:
            arguments = json.loads(tool_call_dict.get("arguments", "{}")) if tool_call_dict.get("arguments") else {}
            tool_calls.append(ToolCall(id=tool_call_dict["id"], tool_name=tool_call_dict["name"], arguments=arguments))
        except json.JSONDecodeError:
            logger.warning(
                "The LLM provider returned a malformed JSON string for tool call arguments. This tool call "
                "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                _id=tool_call_dict["id"],
                _name=tool_call_dict["name"],
                _arguments=tool_call_dict["arguments"],
            )

    # finish_reason can appear in different places so we look for the last one
    finish_reasons = [chunk.finish_reason for chunk in chunks if chunk.finish_reason]
    finish_reason = finish_reasons[-1] if finish_reasons else None

    meta = {
        "model": chunks[-1].meta.get("model"),
        "index": 0,
        "finish_reason": finish_reason,
        "completion_start_time": chunks[0].meta.get("received_at"),  # first chunk received
        "usage": chunks[-1].meta.get("usage"),  # last chunk has the final usage data if available
    }

    return ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, meta=meta)
