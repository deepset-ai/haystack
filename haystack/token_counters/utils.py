# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent
from haystack.dataclasses.chat_message import ChatMessageContentT, ToolCallResultContentT
from haystack.tools import ToolsType, flatten_tools_or_toolsets

# Builds the stand-in for message content that has no text form, such as an image.
_PlaceholderFn = Callable[[ChatMessageContentT], str]


def _non_text_placeholder(content: ChatMessageContentT) -> str:
    """A short stand-in, such as `<image>`, for message content that has no text form."""
    if isinstance(content, ImageContent):
        return "<image>"
    if isinstance(content, FileContent):
        return f"<file: {content.filename or 'unnamed'}>"
    return f"<{type(content).__name__}>"


def _tool_result_text(result: ToolCallResultContentT, placeholder: _PlaceholderFn = _non_text_placeholder) -> str:
    """A tool result as a single string, with placeholders standing in for any non-text parts."""
    if isinstance(result, str):
        return result
    return "".join(block.text if isinstance(block, TextContent) else placeholder(block) for block in result)


def _render_message(
    message: ChatMessage, placeholder: _PlaceholderFn = _non_text_placeholder, include_tool_call_ids: bool = False
) -> str:
    """
    One message as one or more lines of plain text.

    Reasoning content is deliberately left out: providers discard it between turns, so it is not part of the context
    being measured.

    :param message: The message to render.
    :param placeholder: Builds the stand-in for content that has no text form. The default is short and stable, which
        is what a counter needs because the stand-in's own length is what gets measured. A caller rendering for a model
        to read can pass one that describes the content instead.
    :param include_tool_call_ids: Whether to include tool call IDs in the rendered message.
    :returns: The rendered message.
    """
    role = message.role.value
    # A tool-result msg only carries tool_call_results, so it is rendered on its own and labelled with the tool that
    # produced it.
    if results := message.tool_call_results:
        return "\n".join(
            f"[tool:{result.origin.tool_name}"
            f"{' id=' + result.origin.id if include_tool_call_ids and result.origin.id else ''}"
            f"{' (error)' if result.error else ''}] "
            f"{_tool_result_text(result.result, placeholder=placeholder)}"
            for result in results
        )

    lines: list[str] = []
    if texts := message.texts:
        lines.append(f"[{role}] " + "\n".join(texts))
    for call in message.tool_calls:
        arguments = json.dumps(call.arguments, default=str, sort_keys=True)
        call_id = f" id={call.id}" if include_tool_call_ids and call.id else ""
        lines.append(f"[{role} -> tool_call{call_id}] {call.tool_name}({arguments})")
    # Images and files cost tokens too, so they need a stand-in rather than being skipped.
    non_text: list[ChatMessageContentT] = [*message.images, *message.files]
    for content in non_text:
        lines.append(f"[{role}] {placeholder(content)}")
    return "\n".join(lines) if lines else f"[{role}] <no content>"


def _rendered_conversation(
    messages: list[ChatMessage],
    *,
    placeholder: _PlaceholderFn = _non_text_placeholder,
    include_tool_call_ids: bool = False,
) -> str:
    """The whole conversation as one plain-text block, which is what a counter measures."""
    return "\n".join(
        _render_message(message, placeholder=placeholder, include_tool_call_ids=include_tool_call_ids)
        for message in messages
    )


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
