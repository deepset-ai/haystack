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
            if isinstance(tool_call, ChoiceDeltaToolCall):
                tool_call_dict = tool_call.to_dict()
            else:
                tool_call_dict = tool_call

            if tool_call_dict.get("function"):
                # print the tool name
                if tool_call_dict["function"].get("name") and not tool_call_dict["function"].get("arguments"):
                    print("[TOOL CALL]\n", flush=True, end="")
                    print(f"Tool: {tool_call_dict['function']['name']} ", flush=True, end="")

                # print the tool arguments
                if tool_call_dict["function"].get("arguments"):
                    if tool_call_dict["function"]["arguments"].startswith("{"):
                        print("\nArguments: ", flush=True, end="")
                    print(tool_call_dict["function"]["arguments"], flush=True, end="")
                    if tool_call_dict["function"]["arguments"].endswith("}"):
                        print("\n\n", flush=True, end="")

    # Print tool call results if available (from ToolInvoker)
    if chunk.meta.get("tool_result"):
        print(f"[TOOL RESULT]\n{chunk.meta['tool_result']}\n\n", flush=True, end="")

    # Print the main content of the chunk (from ChatGenerator)
    if chunk.content:
        print(chunk.content, flush=True, end="")
