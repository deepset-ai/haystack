# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
    if chunk.start and chunk.index > 0:
        # If this is not the first content block of the message, add two new lines
        print("\n\n", flush=True, end="")

    ## Tool Call streaming
    if chunk.tool_call:
        # Presence of tool_name indicates beginning of a tool call
        # or chunk.tool_call.name: would be equivalent here
        if chunk.start:
            print("[TOOL CALL]\n", flush=True, end="")
            print(f"Tool: {chunk.tool_call.name} ", flush=True, end="")
            print("\nArguments: ", flush=True, end="")

        # print the tool arguments
        if chunk.tool_call.arguments:
            print(chunk.tool_call.arguments, flush=True, end="")

    ## Tool Call Result streaming
    # Print tool call results if available (from ToolInvoker)
    if chunk.tool_call_result:
        # Tool Call Result is fully formed so delta accumulation is not needed
        print(f"[TOOL RESULT]\n{chunk.tool_call_result}", flush=True, end="")

    ## Normal content streaming
    # Print the main content of the chunk (from ChatGenerator)
    if chunk.content:
        if chunk.start:
            print("[ASSISTANT]\n", flush=True, end="")
        print(chunk.content, flush=True, end="")

    # End of LLM assistant message so we add two new lines
    # This ensures spacing between multiple LLM messages (e.g. Agent) or Tool Call Result
    if chunk.meta.get("finish_reason") is not None:
        print("\n\n", flush=True, end="")
