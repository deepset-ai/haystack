# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from haystack.dataclasses import StreamingChunk


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
