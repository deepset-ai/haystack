# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Callable, Optional, Union

from jinja2 import TemplateSyntaxError, nodes
from jinja2.ext import Extension

from haystack import logging
from haystack.dataclasses.chat_message import (
    ChatMessage,
    ChatMessageContentT,
    ChatRole,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolCallResult,
    _deserialize_content_part,
    _serialize_content_part,
)

logger = logging.getLogger(__name__)

START_TAG = "<haystack_content_part>"
END_TAG = "</haystack_content_part>"


class ChatMessageExtension(Extension):
    """
    A Jinja2 extension for creating structured chat messages with mixed content types.

    This extension provides a custom `{% message %}` tag that allows creating chat messages
    with different attributes (role, name, meta) and mixed content types (text, images, etc.).

    Inspired by [Banks](https://github.com/masci/banks).

    Example:
    ```
    {% message role="system" %}
    You are a helpful assistant. You like to talk with {{user_name}}.
    {% endmessage %}

    {% message role="user" %}
    Hello! I am {{user_name}}. Please describe the images.
    {% for image in images %}
    {{ image | templatize_part }}
    {% endfor %}
    {% endmessage %}
    ```

    ### How it works
    1. The `{% message %}` tag is used to define a chat message.
    2. The message can contain text and other structured content parts.
    3. To include a structured content part in the message, the `| templatize_part` filter is used.
       The filter serializes the content part into a JSON string and wraps it in a `<haystack_content_part>` tag.
    4. The `_build_chat_message_json` method of the extension parses the message content parts,
       converts them into a ChatMessage object and serializes it to a JSON string.
    5. The obtained JSON string is usable in the ChatPromptBuilder component, where templates are rendered to actual
       ChatMessage objects.
    """

    SUPPORTED_ROLES = [role.value for role in ChatRole]

    tags = {"message"}

    def parse(self, parser: Any) -> Union[nodes.Node, list[nodes.Node]]:
        """
        Parse the message tag and its attributes in the Jinja2 template.

        This method handles the parsing of role (mandatory), name (optional), meta (optional) and message body content.

        :param parser: The Jinja2 parser instance
        :return: A CallBlock node containing the parsed message configuration
        :raises TemplateSyntaxError: If an invalid role is provided
        """
        lineno = next(parser.stream).lineno

        # Parse role attribute (mandatory)
        parser.stream.expect("name:role")
        parser.stream.expect("assign")
        role_expr = parser.parse_expression()

        if isinstance(role_expr, nodes.Const):
            role = role_expr.value
            if role not in self.SUPPORTED_ROLES:
                raise TemplateSyntaxError(f"Role must be one of: {', '.join(self.SUPPORTED_ROLES)}", lineno)

        # Parse optional name attribute
        name_expr = None
        if parser.stream.current.test("name:name"):
            parser.stream.skip()
            parser.stream.expect("assign")
            name_expr = parser.parse_expression()
            if not isinstance(name_expr.value, str):
                raise TemplateSyntaxError("name must be a string", lineno)

        # Parse optional meta attribute
        meta_expr = None
        if parser.stream.current.test("name:meta"):
            parser.stream.skip()
            parser.stream.expect("assign")
            meta_expr = parser.parse_expression()
            if not isinstance(meta_expr, nodes.Dict):
                raise TemplateSyntaxError("meta must be a dictionary", lineno)

        # Parse message body
        body = parser.parse_statements(("name:endmessage",), drop_needle=True)

        # Build message node with all parameters
        return nodes.CallBlock(
            self.call_method(
                name="_build_chat_message_json",
                args=[role_expr, name_expr or nodes.Const(None), meta_expr or nodes.Dict([])],
            ),
            [],
            [],
            body,
        ).set_lineno(lineno)

    def _build_chat_message_json(self, role: str, name: Optional[str], meta: dict, caller: Callable[[], str]) -> str:
        """
        Build a ChatMessage object from template content and serialize it to a JSON string.

        This method is called by Jinja2 when processing a `{% message %}` tag.
        It takes the rendered content from the template, converts XML blocks into ChatMessageContentT objects,
        creates a ChatMessage object and serializes it to a JSON string.

        :param role: The role of the message
        :param name: Optional name for the message sender
        :param meta: Optional metadata dictionary
        :param caller: Callable that returns the rendered content
        :return: A JSON string representation of the ChatMessage object
        """

        content = caller()
        parts = self._parse_content_parts(content)
        if not parts:
            raise ValueError(
                f"Message template produced content that couldn't be parsed into any message parts. "
                f"Content: '{content!r}'"
            )

        chat_message = self._validate_build_chat_message(parts=parts, role=role, meta=meta, name=name)

        return json.dumps(chat_message.to_dict()) + "\n"

    @staticmethod
    def _parse_content_parts(content: str) -> list[ChatMessageContentT]:
        """
        Parse a string into a sequence of ChatMessageContentT objects.

        This method handles:
        - Plain text content, converted to TextContent objects
        - Structured content parts wrapped in `<haystack_content_part>` tags, converted to ChatMessageContentT objects

        :param content: Input string containing mixed text and content parts
        :return: A list of ChatMessageContentT objects
        :raises ValueError: If the content is empty or contains only whitespace characters or if a
                            `<haystack_content_part>` tag is found without a matching closing tag.
        """
        if not content.strip():
            raise ValueError(
                f"Message content in template is empty or contains only whitespace characters. Content: {content!r}"
            )

        parts: list[ChatMessageContentT] = []
        cursor = 0
        total_length = len(content)

        while cursor < total_length:
            tag_start = content.find(START_TAG, cursor)

            if tag_start == -1:
                # No more tags, add remaining text if any
                remaining_text = content[cursor:].strip()
                if remaining_text:
                    parts.append(TextContent(text=remaining_text))
                break

            # Add text before tag if any
            if tag_start > cursor:
                plain_text = content[cursor:tag_start].strip()
                if plain_text:
                    parts.append(TextContent(text=plain_text))

            content_start = tag_start + len(START_TAG)
            tag_end = content.find(END_TAG, content_start)

            if tag_end == -1:
                raise ValueError(
                    f"Found unclosed <haystack_content_part> tag at position {tag_start}. "
                    f"Content: '{content[tag_start : tag_start + 50]}...'"
                )

            json_content = content[content_start:tag_end]
            data = json.loads(json_content)
            parts.append(_deserialize_content_part(data))

            cursor = tag_end + len(END_TAG)

        return parts

    @staticmethod
    def _validate_build_chat_message(
        parts: list[ChatMessageContentT], role: str, meta: dict, name: Optional[str] = None
    ) -> ChatMessage:
        """
        Validate the parts of a chat message and build a ChatMessage object.

        :param parts: Content parts of the message
        :param role: The role of the message
        :param meta: The metadata of the message
        :param name: The optional name of the message
        :return: A ChatMessage object

        :raises ValueError: If content parts don't allow to build a valid ChatMessage object or the role is not
                            supported
        """

        if role == "user":
            valid_parts = [part for part in parts if isinstance(part, (TextContent, str, ImageContent))]
            if len(parts) != len(valid_parts):
                raise ValueError("User message must contain only TextContent, string or ImageContent parts.")
            return ChatMessage.from_user(meta=meta, name=name, content_parts=valid_parts)

        if role == "system":
            if not isinstance(parts[0], TextContent):
                raise ValueError("System message must contain a text part.")
            text = parts[0].text
            if len(parts) > 1:
                raise ValueError("System message must contain only one text part.")
            return ChatMessage.from_system(meta=meta, name=name, text=text)

        if role == "assistant":
            texts = [part.text for part in parts if isinstance(part, TextContent)]
            tool_calls = [part for part in parts if isinstance(part, ToolCall)]
            reasoning = [part for part in parts if isinstance(part, ReasoningContent)]
            if len(texts) > 1:
                raise ValueError("Assistant message must contain one text part at most.")
            if len(texts) == 0 and len(tool_calls) == 0:
                raise ValueError("Assistant message must contain at least one text or tool call part.")
            if len(parts) > len(texts) + len(tool_calls) + len(reasoning):
                raise ValueError("Assistant message must contain only text, tool call or reasoning parts.")
            return ChatMessage.from_assistant(
                meta=meta,
                name=name,
                text=texts[0] if texts else None,
                tool_calls=tool_calls or None,
                reasoning=reasoning[0] if reasoning else None,
            )

        if role == "tool":
            tool_call_results = [part for part in parts if isinstance(part, ToolCallResult)]
            if len(tool_call_results) == 0 or len(tool_call_results) > 1 or len(parts) > len(tool_call_results):
                raise ValueError("Tool message must contain only one tool call result.")

            tool_result = tool_call_results[0].result
            origin = tool_call_results[0].origin
            error = tool_call_results[0].error

            return ChatMessage.from_tool(meta=meta, tool_result=tool_result, origin=origin, error=error)

        raise ValueError(f"Unsupported role: {role}")


def templatize_part(value: ChatMessageContentT) -> str:
    """
    Jinja filter to convert an ChatMessageContentT object into JSON string wrapped in special XML content tags.

    :param value: The ChatMessageContentT object to convert
    :return: A JSON string wrapped in special XML content tags
    :raises ValueError: If the value is not an instance of ChatMessageContentT
    """
    return f"{START_TAG}{json.dumps(_serialize_content_part(value))}{END_TAG}"
