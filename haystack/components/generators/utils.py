# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
    if chunk.meta.get("tool_calls"):
        for tool_call in chunk.meta["tool_calls"]:
            if isinstance(tool_call, ChoiceDeltaToolCall) and tool_call.function:
                # print the tool name
                if tool_call.function.name and not tool_call.function.arguments:
                    print("[TOOL CALL]\n", flush=True, end="")
                    print(f"Tool: {tool_call.function.name} ", flush=True, end="")
                    print("\nArguments: ", flush=True, end="")

                # print the tool arguments
                if tool_call.function.arguments:
                    print(tool_call.function.arguments, flush=True, end="")

    # Print tool call results if available (from ToolInvoker)
    if chunk.meta.get("tool_result"):
        print(f"\n\n[TOOL RESULT]\n{chunk.meta['tool_result']}\n\n", flush=True, end="")

    # Print the main content of the chunk (from ChatGenerator)
    if chunk.content:
        print(chunk.content, flush=True, end="")
