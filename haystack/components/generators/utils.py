# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from haystack.dataclasses import ChatMessage, StreamingChunk


def print_streaming_chunk(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.

    Prints the tokens of the first completion to stdout as soon as they are received
    """
    if chunk.meta.get("tool_calls"):
        for tool_call in chunk.meta["tool_calls"]:
            if isinstance(tool_call, ChoiceDeltaToolCall):
                if tool_call.function.name:
                    print(tool_call.function.name, flush=True, end="")
                elif tool_call.function.arguments:
                    print(tool_call.function.arguments, flush=True, end="")

    print(chunk.content, flush=True, end="")
