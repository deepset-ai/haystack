# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict
from typing import Any, Callable, List, Optional, Union

from jinja2 import Environment, TemplateSyntaxError, nodes
from jinja2.ext import Extension

from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ImageContent, TextContent
from haystack.lazy_imports import LazyImport

with LazyImport(message='Run "pip install arrow>=1.3.0"') as arrow_import:
    import arrow


class Jinja2TimeExtension(Extension):
    # Syntax for current date
    tags = {"now"}

    def __init__(self, environment: Environment):  # pylint: disable=useless-parent-delegation
        """
        Initializes the JinjaTimeExtension object.

        :param environment: The Jinja2 environment to initialize the extension with.
            It provides the context where the extension will operate.
        """
        arrow_import.check()
        super().__init__(environment)

    @staticmethod
    def _get_datetime(
        timezone: str,
        operator: Optional[str] = None,
        offset: Optional[str] = None,
        datetime_format: Optional[str] = None,
    ) -> str:
        """
        Get the current datetime based on timezone, apply any offset if provided, and format the result.

        :param timezone: The timezone string (e.g., 'UTC' or 'America/New_York') for which the current
            time should be fetched.
        :param operator: The operator ('+' or '-') to apply to the offset (used for adding/subtracting intervals).
            Defaults to None if no offset is applied, otherwise default is '+'.
        :param offset: The offset string in the format 'interval=value' (e.g., 'hours=2,days=1') specifying how much
            to adjust the datetime. The intervals can be any valid interval accepted
            by Arrow (e.g., hours, days, weeks, months). Defaults to None if no adjustment is needed.
        :param datetime_format: The format string to use for formatting the output datetime.
            Defaults to '%Y-%m-%d %H:%M:%S' if not provided.
        """
        try:
            dt = arrow.now(timezone)
        except Exception as e:
            raise ValueError(f"Invalid timezone {timezone}: {e}")

        if offset and operator:
            try:
                # Parse the offset and apply it to the datetime object
                replace_params = {
                    interval.strip(): float(operator + value.strip())
                    for param in offset.split(",")
                    for interval, value in [param.split("=")]
                }
                # Shift the datetime fields based on the parsed offset
                dt = dt.shift(**replace_params)
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid offset or operator {offset}, {operator}: {e}")

        # Use the provided format or fallback to the default one
        datetime_format = datetime_format or "%Y-%m-%d %H:%M:%S"

        return dt.strftime(datetime_format)

    def parse(self, parser: Any) -> Union[nodes.Node, List[nodes.Node]]:
        """
        Parse the template expression to determine how to handle the datetime formatting.

        :param parser: The parser object that processes the template expressions and manages the syntax tree.
            It's used to interpret the template's structure.
        """
        lineno = next(parser.stream).lineno
        node = parser.parse_expression()
        # Check if a custom datetime format is provided after a comma
        datetime_format = parser.parse_expression() if parser.stream.skip_if("comma") else nodes.Const(None)

        # Default Add when no operator is provided
        operator = "+" if isinstance(node, nodes.Add) else "-"
        # Call the _get_datetime method with the appropriate operator and offset, if exist
        call_method = self.call_method(
            "_get_datetime",
            [node.left, nodes.Const(operator), node.right, datetime_format]
            if isinstance(node, (nodes.Add, nodes.Sub))
            else [node, nodes.Const(None), nodes.Const(None), datetime_format],
            lineno=lineno,
        )

        return nodes.Output([call_method], lineno=lineno)


class ChatMessageExtension(Extension):
    """
    A Jinja2 extension for creating structured chat messages with mixed content types.

    This extension provides a custom {% message %} tag that allows creating chat messages
    with different roles (system, user) and mixed content types (text, images).

    Inspired by [Banks](https://github.com/masci/banks).

    Example:
    {% message role="system" %}
    You are a helpful assistant. You like to talk with {{user_name}}.
    {% endmessage %}

    {% message role="user" %}
    Hello! I am {{user_name}}. Please describe the images.
    {% for image in images %}
    {{ image | for_template }}
    {% endfor %}
    {% endmessage %}
    """

    SUPPORTED_ROLES = ("system", "user")

    tags = {"message"}

    def parse(self, parser):
        """
        Jinja2 extension parse method.
        """
        lineno = next(parser.stream).lineno

        # Parse role attribute
        parser.stream.expect("name:role")
        parser.stream.expect("assign")
        role_value = parser.parse_expression()

        role = role_value.value
        if role not in self.SUPPORTED_ROLES:
            raise TemplateSyntaxError(f"Role must be one of: {', '.join(self.SUPPORTED_ROLES)}", lineno)

        # Parse message body
        body = parser.parse_statements(("name:endmessage",), drop_needle=True)

        # Build message node
        return nodes.CallBlock(
            self.call_method("_build_chat_message_json", [nodes.Const(role)]), [], [], body
        ).set_lineno(lineno)

    def _build_chat_message_json(self, role: str, caller: Callable[[], str]) -> str:
        """
        Build a chat message from template content and convert it to JSON.

        This method is called by Jinja2 when processing a {% message %} tag.
        It takes the rendered content from the template, parses it into content parts,
        and converts the resulting ChatMessage into a JSON string.
        """
        content = caller()
        parts = self._parse_content_parts(content)
        if not parts:
            parts = [TextContent(text=content)]
        message = ChatMessage(_role=ChatRole(role), _content=parts)
        return json.dumps(message.to_dict()) + "\n"

    def _parse_content_parts(self, content: str) -> List[Union[TextContent, ImageContent]]:
        """
        Parse a string into a sequence of chat message content parts.

        The input may contain both plain text and special content parts (like images) marked with XML-like tags.
        """
        parts: list[Union[TextContent, ImageContent]] = []
        text_buffer = ""

        segments = content.split("<haystack_content_part>")

        # First segment is always text (if not empty)
        if segments[0].strip():
            parts.append(TextContent(text=segments[0].strip()))

        # Process remaining segments
        for segment in segments[1:]:
            # Split at end tag
            if "</haystack_content_part>" not in segment:
                text_buffer += segment
                continue

            content_part, remaining_text = segment.split("</haystack_content_part>", 1)

            # Parse content part
            try:
                data = json.loads(content_part)
                # currently we only support images but this should be made more generic
                if "base64_image" in data:
                    parts.append(ImageContent(**data))
            except json.JSONDecodeError:
                text_buffer += content_part

            # Add any text after the content part
            if remaining_text.strip():
                parts.append(TextContent(text=remaining_text.strip()))

        return parts


def for_template(value: ImageContent) -> str:
    """
    Convert an ImageContent object into a template-safe string representation.

    This filter wraps the image content in special tags that can be parsed back
    into an ImageContent object when the template is rendered.
    """
    if not isinstance(value, ImageContent):
        raise ValueError("Value must be an instance of ImageContent")
    return f"<haystack_content_part>{json.dumps(asdict(value))}</haystack_content_part>"
