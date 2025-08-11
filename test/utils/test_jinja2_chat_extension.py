# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

import pytest
from jinja2 import TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment

from haystack.dataclasses.chat_message import ImageContent, ReasoningContent, ToolCall, ToolCallResult
from haystack.utils.jinja2_chat_extension import ChatMessageExtension, templatize_part


class TestChatMessageExtension:
    @pytest.fixture
    def jinja_env(self) -> SandboxedEnvironment:
        # we use a SandboxedEnvironment here to replicate the conditions of the ChatPromptBuilder component
        env = SandboxedEnvironment(extensions=[ChatMessageExtension])
        env.filters["templatize_part"] = templatize_part
        return env

    def test_message_with_name_and_meta(self, jinja_env):
        template = """
        {% message role="user" name="Bob" meta={"language": "en"} %}
        Hello!
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {"role": "user", "name": "Bob", "content": [{"text": "Hello!"}], "meta": {"language": "en"}}
        assert output == expected

    def test_message_no_endmessage_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        Hello!
        """
        with pytest.raises(TemplateSyntaxError, match="Jinja was looking for the following tags: 'endmessage'"):
            jinja_env.from_string(template).render()

    def test_message_no_role_raises_error(self, jinja_env):
        template = """
        {% message %}
        You are a helpful assistant.
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError, match="expected token 'role'"):
            jinja_env.from_string(template).render()

    def test_message_meta_not_dict_raises_error(self, jinja_env):
        template = """
        {% message role="user" meta="not a dict" %}
        Hello!
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError, match="meta must be a dictionary"):
            jinja_env.from_string(template).render()

    def test_message_meta_invalid_dict_raises_error(self, jinja_env):
        template = """
        {% message role="user" meta={"key": "unclosed_value} %}
        Hello!
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError):
            jinja_env.from_string(template).render()

    def test_message_name_not_str_raises_error(self, jinja_env):
        template = """
        {% message role="user" name=123 %}
        Hello!
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError, match="name must be a string"):
            jinja_env.from_string(template).render()

    def test_system_message(self, jinja_env):
        template = """
        {% message role="system" %}
        You are a helpful assistant.
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {"role": "system", "content": [{"text": "You are a helpful assistant."}], "name": None, "meta": {}}
        assert output == expected

    def test_user_message_with_variable(self, jinja_env):
        template = """
        {% message role="user" %}
        Hello, my name is {{name}}!
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render(name="Alice")
        output = json.loads(rendered.strip())
        expected = {"role": "user", "content": [{"text": "Hello, my name is Alice!"}], "name": None, "meta": {}}
        assert output == expected

    def test_assistant_message_with_tool_call(self, jinja_env):
        template = """
        {% message role="assistant" %}
        Let me search for that information.
        {{ tool_call | templatize_part }}
        {% endmessage %}
        """
        tool_call = ToolCall(tool_name="search", arguments={"query": "an interesting question"}, id="search_1")
        rendered = jinja_env.from_string(template).render(tool_call=tool_call)
        output = json.loads(rendered.strip())
        expected = {
            "role": "assistant",
            "content": [
                {"text": "Let me search for that information."},
                {
                    "tool_call": {
                        "tool_name": "search",
                        "arguments": {"query": "an interesting question"},
                        "id": "search_1",
                    }
                },
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_assistant_message_with_reasoning(self, jinja_env):
        template = """
        {% message role="assistant" %}
        {{ reasoning | templatize_part }}
        The answer is 4.
        {% endmessage %}
        """
        reasoning = ReasoningContent(reasoning_text="Let me think about it...", extra={"key": "value"})
        rendered = jinja_env.from_string(template).render(reasoning=reasoning)
        output = json.loads(rendered.strip())
        expected = {
            "role": "assistant",
            "content": [
                {"reasoning": {"reasoning_text": "Let me think about it...", "extra": {"key": "value"}}},
                {"text": "The answer is 4."},
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_tool_message(self, jinja_env):
        template = """
        {% message role="tool" %}
        {{ tool_result | templatize_part }}
        {% endmessage %}
        """
        tool_call = ToolCall(tool_name="search", arguments={"query": "test"}, id="search_1")
        tool_result = ToolCallResult(result="Here are the search results", origin=tool_call, error=False)
        rendered = jinja_env.from_string(template).render(tool_result=tool_result)
        output = json.loads(rendered.strip())
        expected = {
            "role": "tool",
            "content": [
                {
                    "tool_call_result": {
                        "result": "Here are the search results",
                        "error": False,
                        "origin": {"tool_name": "search", "arguments": {"query": "test"}, "id": "search_1"},
                    }
                }
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_user_message_with_image(self, jinja_env, base64_image_string):
        template = """
        {% message role="user" %}
        Please describe this image:
        {{ image | templatize_part }}
        {% endmessage %}
        """
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        rendered = jinja_env.from_string(template).render(image=image)
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {"text": "Please describe this image:"},
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": None,
                        "meta": {},
                        "validation": True,
                    }
                },
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_user_message_with_multiple_images(self, jinja_env, base64_image_string):
        template = """
        {% message role="user" %}
        Compare these images:
        {% for img in images %}
        {{ img | templatize_part }}
        {% endfor %}
        {% endmessage %}
        """
        images = [
            ImageContent(base64_image=base64_image_string, mime_type="image/png"),
            ImageContent(base64_image=base64_image_string, mime_type="image/png"),
        ]
        rendered = jinja_env.from_string(template).render(images=images)
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {"text": "Compare these images:"},
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": None,
                        "meta": {},
                        "validation": True,
                    }
                },
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": None,
                        "meta": {},
                        "validation": True,
                    }
                },
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_user_message_with_multiple_images_and_interleaved_text(self, jinja_env, base64_image_string):
        """
        Tests that messages with multiple images and interleaved text are rendered correctly.
        This format is used by Anthropic models:
        https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images
        """
        template = """
        {% message role="user" %}
        {% for image in images %}
        Image {{ loop.index }}:
        {{ image | templatize_part }}
        {% endfor %}
        What's the difference between the two images?
        {% endmessage %}
        """
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        rendered = jinja_env.from_string(template).render(images=[image, image])
        output = json.loads(rendered.strip())

        expected = {
            "role": "user",
            "content": [
                {"text": "Image 1:"},
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": None,
                        "meta": {},
                        "validation": True,
                    }
                },
                {"text": "Image 2:"},
                {
                    "image": {
                        "base64_image": base64_image_string,
                        "mime_type": "image/png",
                        "detail": None,
                        "meta": {},
                        "validation": True,
                    }
                },
                {"text": "What's the difference between the two images?"},
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_user_message_multiple_lines(self, jinja_env):
        template = """
{% message role="user" %}
What do you think of NLP?
It's an interesting domain, if you ask me.
But my favorite subject is Small Language Models.
{% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {
                    "text": (
                        "What do you think of NLP?\nIt's an interesting domain, if you ask me.\n"
                        "But my favorite subject is Small Language Models."
                    )
                }
            ],
            "name": None,
            "meta": {},
        }
        assert output == expected

    def test_invalid_role(self, jinja_env):
        template = """
        {% message role="invalid_role" %}
        This should fail
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError, match="Role must be one of"):
            jinja_env.from_string(template).render()

    def test_templatize_part_filter_with_invalid_type(self):
        with pytest.raises(TypeError, match="Unsupported type in ChatMessage content"):
            templatize_part(123)

    def test_empty_message_content_raises_error(self, jinja_env):
        error_message = "Message content in template is empty or contains only whitespace characters."

        template = "{% message role='user' %}{% endmessage %}"
        with pytest.raises(ValueError, match=error_message):
            jinja_env.from_string(template).render()

        template = "{% message role='user' %}  \n\t\n\t  {% endmessage %}"
        with pytest.raises(ValueError, match=error_message):
            jinja_env.from_string(template).render()

        template = """
        {% message role="user" %}
        {% if some_condition %}
        {% else %}
        {% endif %}
        {% endmessage %}
        """
        with pytest.raises(ValueError, match=error_message):
            jinja_env.from_string(template).render()

        template = """
        {% message role="user" %}
        {{ variable_that_doesnt_exist }}
        {% endmessage %}
        """
        with pytest.raises(ValueError, match=error_message):
            jinja_env.from_string(template).render()

    def test_message_no_parts_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        Something
        {% endmessage %}
        """

        # I am patching the _parse_content_parts method to return an empty list
        # because I could not find a template that raises a similar error
        with patch("haystack.utils.jinja2_chat_extension.ChatMessageExtension._parse_content_parts", return_value=[]):
            with pytest.raises(ValueError, match="message parts"):
                jinja_env.from_string(template).render()

    def test_message_with_whitespace_handling(self, jinja_env):
        # the following templates should all be equivalent
        templates = [
            """{% message role="user" %}String{% endmessage %}""",
            """{% message role="user" %}    String    {% endmessage %}""",
            """{% message role="user" %}
            String
            {% endmessage %}""",
            """{% message role="user" %}\tString\t{% endmessage %}""",
        ]
        expected = {"role": "user", "content": [{"text": "String"}], "name": None, "meta": {}}
        for template in templates:
            rendered = jinja_env.from_string(template).render()
            output = json.loads(rendered.strip())
            assert output == expected

    def test_unclosed_content_tag_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        <haystack_content_part>{"type": "text", "text": "Hello"}
        {% endmessage %}
        """
        with pytest.raises(ValueError, match="Found unclosed <haystack_content_part> tag"):
            jinja_env.from_string(template).render()

    def test_invalid_json_in_content_part_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        Normal text before.
        <haystack_content_part>{"this is": "invalid" json}</haystack_content_part>
        <haystack_content_part>not even trying to be json</haystack_content_part>
        <haystack_content_part>{]</haystack_content_part>
        Normal text after.
        {% endmessage %}
        """
        with pytest.raises(json.JSONDecodeError):
            jinja_env.from_string(template).render()

    def test_invalid_system_message_raises_error(self, jinja_env, base64_image_string):
        template = """
        {% message role="system" %}
        {{ image | templatize_part }}
        {% endmessage %}
        """
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

        template = """
        {% message role="system" %}
        Some text.
        {{ image | templatize_part }}
        {% endmessage %}
        """
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

    def test_invalid_assistant_message_raises_error(self, jinja_env, base64_image_string):
        template = """
        {% message role="assistant" %}
        text 1
        {{ image | templatize_part }}
        text 2
        {% endmessage %}
        """
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

        template = """
        {% message role="assistant" %}
        {{ image | templatize_part }}
        {% endmessage %}
        """
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

        template = """
        {% message role="assistant" %}
        text 1
        {{ image | templatize_part }}
        {% endmessage %}
        """
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

    def test_invalid_tool_message_raises_error(self, jinja_env, base64_image_string):
        template = """
        {% message role="tool" %}
        {{ image | templatize_part }}
        {% endmessage %}
        """
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(image=image)

        template = """
        {% message role="tool" %}
        {{ tool_result | templatize_part }}
        {{ tool_result | templatize_part }}
        {% endmessage %}
        """
        tool_call = ToolCall(tool_name="search", arguments={"query": "test"}, id="search_1")
        tool_result = ToolCallResult(result="Here are the search results", origin=tool_call, error=False)
        with pytest.raises(ValueError):
            jinja_env.from_string(template).render(tool_result=tool_result)

        template = """
        {% message role="tool" %}
        {{ tool_result | templatize_part }}
        {{ image | templatize_part }}
        {% endmessage %}
        """
        with pytest.raises(TypeError):
            jinja_env.from_string(template).render(image=image)
