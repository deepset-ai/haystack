# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent
from haystack.dataclasses.chat_message import ChatMessageContentT, ToolCallResultContentT
from haystack.tools import ToolsType, flatten_tools_or_toolsets


def _non_text_placeholder(content: ChatMessageContentT) -> str:
    """A short stand-in, such as `<image>`, for message content that has no text form."""
    if isinstance(content, ImageContent):
        return "<image>"
    if isinstance(content, FileContent):
        return f"<file: {content.filename or 'unnamed'}>"
    return f"<{type(content).__name__}>"


def _tool_result_text(result: ToolCallResultContentT) -> str:
    """A tool result as a single string, with placeholders standing in for any non-text parts."""
    if isinstance(result, str):
        return result
    return "".join(block.text if isinstance(block, TextContent) else _non_text_placeholder(block) for block in result)


def _render_message(message: ChatMessage) -> str:
    """
    One message as one or more lines of plain text.

    Reasoning content is deliberately left out: providers discard it between turns, so it is not part of the context
    being measured.
    """
    role = message.role.value
    # A tool-result msg only carries tool_call_results, so it is rendered on its own and labelled with the tool that
    # produced it.
    if results := message.tool_call_results:
        return "\n".join(
            f"[tool:{result.origin.tool_name}{' (error)' if result.error else ''}] {_tool_result_text(result.result)}"
            for result in results
        )

    lines: list[str] = []
    if texts := message.texts:
        lines.append(f"[{role}] " + "\n".join(texts))
    for call in message.tool_calls:
        arguments = json.dumps(call.arguments, default=str, sort_keys=True)
        lines.append(f"[{role} -> tool_call] {call.tool_name}({arguments})")
    # Images and files cost tokens too, so they need a stand-in rather than being skipped.
    non_text: list[ChatMessageContentT] = [*message.images, *message.files]
    for content in non_text:
        lines.append(f"[{role}] {_non_text_placeholder(content)}")
    return "\n".join(lines) if lines else f"[{role}] <no content>"


def _rendered_conversation(messages: list[ChatMessage]) -> str:
    """The whole conversation as one plain-text block, which is what a counter measures."""
    return "\n".join(_render_message(message) for message in messages)


def _rendered_tools(tools: ToolsType | None) -> str:
    """
    Tool schemas as one JSON block, standing in for what a provider sends alongside the messages.

    :param tools: The tools to render, or None.
    :returns: The rendered schemas, or an empty string when there are none.
    """
    if not tools:
        return ""
    return json.dumps([tool.tool_spec for tool in flatten_tools_or_toolsets(tools)], default=str, sort_keys=True)


def _non_text_tokens(messages: list[ChatMessage], *, tokens_per_image: int, tokens_per_file: int) -> int:
    """
    A flat estimate for the images and files in a conversation, which have no text to measure.

    :param messages: The conversation to measure.
    :param tokens_per_image: Tokens to charge per image.
    :param tokens_per_file: Tokens to charge per file.
    :returns: The estimated token count for the non-text content.
    """
    images = 0
    files = 0
    for message in messages:
        images += len(message.images)
        files += len(message.files)
        # Images and files returned by a tool are nested inside the tool result
        for result in message.tool_call_results:
            if isinstance(result.result, str):
                continue
            images += sum(isinstance(block, ImageContent) for block in result.result)
            files += sum(isinstance(block, FileContent) for block in result.result)
    return images * tokens_per_image + files * tokens_per_file
