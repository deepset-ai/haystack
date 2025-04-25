# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from haystack.dataclasses import ChatMessage, StreamingCallbackT, StreamingChunk


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.

    Prints the tokens of the first completion to stdout as soon as they are received
    """
    print(chunk.content, flush=True, end="")


def _emit_tool_call_info(message: ChatMessage) -> None:
    """
    Emit information about a tool call including the tool name and arguments.

    :param message: The message containing the tool call
    """
    if message.tool_call is None:
        return

    # Create a chunk with tool call information
    tool_call_info = f"Tool Call: {message.tool_call.tool_name} "  # Note the space after tool name
    if message.tool_call.arguments:
        # Pre-format arguments string for better readability
        args_str = ", ".join(f"{k}={v}" for k, v in message.tool_call.arguments.items())
        tool_call_info += f"({args_str})\n"  # Note the newline at the end
    else:
        tool_call_info += "\n"  # Always add newline for consistency

    print(tool_call_info, flush=True, end="")
