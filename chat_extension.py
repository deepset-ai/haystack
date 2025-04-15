import json
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union, contextlib

from jinja2 import TemplateSyntaxError, nodes
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment

from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ImageContent, TextContent

SUPPORTED_ROLES = ("system", "user")


class ChatMessageExtension(Extension):
    """
    A Jinja2 extension for creating structured chat messages with mixed content types.

    This extension provides a custom {% message %} tag that allows creating chat messages
    with different roles (system, user) and mixed content types (text, images).

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
        if role not in SUPPORTED_ROLES:
            raise TemplateSyntaxError(f"Role must be one of: {', '.join(SUPPORTED_ROLES)}", lineno)

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


def render_chat_messages(template, data: Dict[str, Any] = None) -> List[ChatMessage]:
    """
    This should be part of the ChatPromptBuilder.
    """
    data = data or {}
    rendered = template.render(data)

    print(f"rendered: {rendered}")
    print("--------------------------------")

    messages = []
    for line in rendered.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        try:
            messages.append(ChatMessage.from_dict(json.loads(line)))
        except Exception:
            contextlib.suppress(Exception)

    # Fallback for templates without message tags
    if not messages:
        content = rendered.strip()
        if content:
            messages.append(ChatMessage(_role=ChatRole("user"), _content=[TextContent(text=content)]))

    return messages


img = ImageContent(base64_image="something in base64", mime_type="image/png")

env = SandboxedEnvironment(extensions=[ChatMessageExtension])
env.filters["for_template"] = for_template

compiled_template = env.from_string("""
{% message role="system" %}
You are a helpful assistant. You like to talk with {{ name }}.
{% endmessage %}

{% message role="user" %}
Hello, I am {{ name }}. Can you please describe the images?
{% for image in images %}
{{ image | for_template }}
{% endfor %}
{% endmessage %}
""")

print(render_chat_messages(compiled_template, {"name": "John", "images": [img, img]}))
