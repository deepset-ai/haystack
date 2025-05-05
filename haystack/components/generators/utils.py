# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from haystack.dataclasses import StreamingChunk


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.

    Prints the tokens of the first completion to stdout as soon as they are received.
    Also prints the tool calls in the meta data of the chunk.
    """
    if chunk.meta.get("tool_calls"):
        for tool_call in chunk.meta["tool_calls"]:
            if isinstance(tool_call, ChoiceDeltaToolCall) and tool_call.function:
                if tool_call.function.name and not tool_call.function.arguments:
                    print(f"[TOOL CALL - {tool_call.function.name}] ", flush=True, end="")

                if tool_call.function.arguments:
                    if tool_call.function.arguments.startswith("{"):
                        print("\nArguments: ", flush=True, end="")
                    print(tool_call.function.arguments, flush=True, end="")
                    if tool_call.function.arguments.endswith("}"):
                        print("\n\n", flush=True, end="")

    if chunk.meta.get("tool_result"):
        print(f"[TOOL RESULT]\n{chunk.meta['tool_result']}\n\n", flush=True, end="")

    if chunk.content:
        print(chunk.content, flush=True, end="")
