# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.dataclasses import ChatMessage, ImageContent, ReasoningContent, TextContent
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install \"transformers[torch]\"'") as torch_import:
    import torch

logger = logging.getLogger(__name__)


def serialize_hf_model_kwargs(kwargs: dict[str, Any]) -> None:
    """
    Recursively serialize HuggingFace specific model keyword arguments in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to serialize
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, torch.dtype):
            kwargs[k] = str(v)

        if isinstance(v, dict):
            serialize_hf_model_kwargs(v)


def deserialize_hf_model_kwargs(kwargs: dict[str, Any]) -> None:
    """
    Recursively deserialize HuggingFace specific model keyword arguments in-place to make them JSON serializable.

    :param kwargs: The keyword arguments to deserialize
    """
    torch_import.check()

    for k, v in kwargs.items():
        # torch.dtype
        if isinstance(v, str) and v.startswith("torch."):
            dtype_str = v.split(".")[1]
            dtype = getattr(torch, dtype_str, None)
            if dtype is not None and isinstance(dtype, torch.dtype):
                kwargs[k] = dtype

        if isinstance(v, dict):
            deserialize_hf_model_kwargs(v)


def convert_message_to_hf_format(message: ChatMessage) -> dict[str, Any]:
    """
    Convert a message to the format expected by Hugging Face.

    Note: ReasoningContent is skipped during conversion because the HuggingFace Inference API
    (which follows the OpenAI-compatible chat completion format) does not support reasoning
    in input messages. Reasoning is captured from model outputs for transparency but is not
    sent back to the API in multi-turn conversations.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results
    images = message.images

    # Filter out ReasoningContent from the content list for validation
    # ReasoningContent is for human transparency only, not sent to the API
    non_reasoning_content = [c for c in message._content if not isinstance(c, ReasoningContent)]

    if not text_contents and not tool_calls and not tool_call_results and not images:
        raise ValueError(
            "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, `ToolCallResult`, or `ImageContent`."
        )
    if len(tool_call_results) > 0 and len(non_reasoning_content) > 1:
        raise ValueError(
            "For compatibility with the Hugging Face API, a `ChatMessage` with a `ToolCallResult` "
            "cannot contain any other content."
        )

    # HF always expects a content field, even if it is empty
    hf_msg: dict[str, Any] = {"role": message._role.value, "content": ""}

    if tool_call_results:
        result = tool_call_results[0]
        hf_msg["content"] = result.result
        if tc_id := result.origin.id:
            hf_msg["tool_call_id"] = tc_id
        # HF does not provide a way to communicate errors in tool invocations, so we ignore the error field
        return hf_msg

    # Handle multimodal content (text + images) preserving order
    if text_contents or images:
        content_parts: list[dict[str, Any]] = []
        for part in message._content:
            if isinstance(part, TextContent):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                image_url = f"data:{part.mime_type or 'image/jpeg'};base64,{part.base64_image}"
                content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

        if len(content_parts) == 1 and not images:
            # content is a string
            hf_msg["content"] = content_parts[0]["text"]
        else:
            hf_msg["content"] = content_parts

    if tool_calls:
        hf_tool_calls = []
        for tc in tool_calls:
            hf_tool_call = {"type": "function", "function": {"name": tc.tool_name, "arguments": tc.arguments}}
            if tc.id is not None:
                hf_tool_call["id"] = tc.id
            hf_tool_calls.append(hf_tool_call)
        hf_msg["tool_calls"] = hf_tool_calls

    return hf_msg
