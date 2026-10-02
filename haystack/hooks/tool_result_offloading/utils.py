# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
from pathlib import Path
from typing import Any

from haystack import logging
from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent
from haystack.dataclasses.chat_message import ToolCallResult
from haystack.hooks.tool_result_offloading.types import ToolResultStore

logger = logging.getLogger(__name__)

# Meta key marking an already-offloaded tool-result message; its value is the list of store references written.
# Stops a second offload from offloading the pointer text again, since the pointer is itself a tool result.
_OFFLOADED_META_KEY = "tool_result_offloaded"

# Extension used for a binary block whose MIME type is unknown or maps to no known extension.
_FALLBACK_EXTENSION = ".bin"


def _content_block_payload(content_block: TextContent | ImageContent | FileContent) -> str:
    """
    Return the string a content block contributes to the conversation.

    For an image or a file this is the base64 payload, which is what actually occupies the context window.

    :param content_block: The content block to inspect.
    :returns: The content block's text or base64 payload.
    """
    if isinstance(content_block, TextContent):
        return content_block.text
    return content_block.base64_image if isinstance(content_block, ImageContent) else content_block.base64_data


def _offloadable_content_blocks(
    result: ToolCallResult, store: ToolResultStore
) -> list[TextContent | ImageContent | FileContent] | None:
    """
    Return a result's content blocks when `store` can take them, or None when the result stays in context.

    A result made up of nothing but empty text has nothing worth storing. Image and file content only goes to a store
    that sets `supports_binary_content`; a text-only store leaves the whole result in context and logs a warning.

    :param result: The tool result to inspect.
    :param store: The store the result would be written to.
    :returns: The result's content blocks, or None when the result cannot or should not be offloaded.
    """
    # A plain string result is handled as a single text block, so callers have one shape to work with.
    content_blocks: list[TextContent | ImageContent | FileContent] = (
        [TextContent(text=result.result)] if isinstance(result.result, str) else list(result.result)
    )

    # `all` also covers a result with no content blocks at all.
    if all(isinstance(content_block, TextContent) and not content_block.text for content_block in content_blocks):
        return None

    if not getattr(store, "supports_binary_content", False) and not all(
        isinstance(content_block, TextContent) for content_block in content_blocks
    ):
        logger.warning(
            "Tool '{tool}' produced a result with image or file content, but {store} does not support binary "
            "content; leaving the result in context.",
            tool=result.origin.tool_name,
            store=type(store).__name__,
        )
        return None

    return content_blocks


def _offloaded_message(
    message: ChatMessage,
    *,
    content_blocks: list[TextContent | ImageContent | FileContent],
    store: ToolResultStore,
    key_prefix: str,
    preview_chars: int,
    additional_meta: dict[str, Any] | None = None,
) -> ChatMessage:
    """
    Write a tool result to the store and return the message that points to it.

    Callers decide whether a result should be offloaded; keeping the write and the message construction here gives
    every offloading entry point the same pointer format and metadata marker.

    :param message: The tool-result message being offloaded. Must carry a tool result.
    :param content_blocks: The result's content blocks, as returned by `_offloadable_content_blocks`.
    :param store: The store to write the content blocks to.
    :param key_prefix: The result's store key prefix, unique within the run so results from different tools and steps
        do not collide.
    :param preview_chars: Number of leading characters of each offloaded text to include in the pointer.
    :param additional_meta: Extra metadata to record on the offloaded message, such as a compaction marker.
    :returns: A new tool-result message whose content points at the stored result.
    """
    result = message.tool_call_result
    if result is None:
        raise ValueError("Only tool-result messages can be offloaded.")

    references, pointer = _offload_content_blocks(
        content_blocks=content_blocks, store=store, prefix=key_prefix, preview_chars=preview_chars
    )
    return ChatMessage.from_tool(
        tool_result=pointer,
        origin=result.origin,
        error=result.error,
        meta={**message.meta, **(additional_meta or {}), _OFFLOADED_META_KEY: references},
    )


def _offload_content_blocks(
    content_blocks: list[TextContent | ImageContent | FileContent],
    store: ToolResultStore,
    prefix: str,
    preview_chars: int,
) -> tuple[list[str], str]:
    """
    Write a result's content blocks to the store and build the pointer that replaces them in the conversation.

    Every content block goes to its own store entry. A single block keeps `prefix` as its key and gets a one-line
    pointer; several blocks get position-suffixed keys and one numbered pointer line each.

    :param content_blocks: The result's content blocks, in order.
    :param store: The store to write to.
    :param prefix: The result's store key prefix, as described above.
    :param preview_chars: Number of leading characters of each offloaded text to include in the pointer.
    :returns: The store references written, and the pointer text for the conversation.
    """
    references: list[str] = []
    descriptions: list[str] = []
    single = len(content_blocks) == 1
    for position, content_block in enumerate(content_blocks):
        reference, description = _offload_content_block(
            content_block=content_block,
            store=store,
            key_prefix=prefix if single else f"{prefix}_{position}",
            preview_chars=preview_chars,
        )
        references.append(reference)
        descriptions.append(description)

    if len(descriptions) == 1:
        return references, f"Tool result offloaded to {descriptions[0]}"

    numbered = [f"{position}. {description}" for position, description in enumerate(descriptions, start=1)]
    return references, "\n".join([f"Tool result offloaded to {len(descriptions)} files:", *numbered])


def _offload_content_block(
    content_block: TextContent | ImageContent | FileContent, store: ToolResultStore, key_prefix: str, preview_chars: int
) -> tuple[str, str]:
    """
    Write a single content block to the store and describe where it went.

    :param content_block: The content block to offload.
    :param store: The store to write to.
    :param key_prefix: The content block's store key without its extension.
    :param preview_chars: Number of leading characters of an offloaded text to include in its description.
    :returns: The store reference the content block was written to, and a one-line description for the pointer.
    """
    if isinstance(content_block, TextContent):
        text = content_block.text
        reference = store.write(key=f"{key_prefix}.txt", content=text)
        # An ellipsis marks a preview that was cut short, so the model can tell it is not the whole text.
        preview = f"{text[:preview_chars]}{'...' if len(text) > preview_chars else ''}"
        return reference, f"text ({len(text)} characters) at '{reference}'. Preview: {preview}"

    if isinstance(content_block, ImageContent):
        data = base64.b64decode(content_block.base64_image)
        label = content_block.mime_type or "image"
        filename = None
    else:
        data = base64.b64decode(content_block.base64_data)
        label = content_block.mime_type or "file"
        filename = content_block.filename

    # What the tool called the file wins over its MIME type. Only the suffix is taken.
    mime_extension = mimetypes.guess_extension(content_block.mime_type) if content_block.mime_type else None
    extension = (Path(filename).suffix if filename else "") or mime_extension or _FALLBACK_EXTENSION
    reference = store.write(key=f"{key_prefix}{extension}", content=data)

    named = f" named '{filename}'" if filename else ""
    return reference, f"{label}{named} ({len(data)} bytes) at '{reference}'"
