# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import MagicMock

import pytest

from haystack.dataclasses import ChatMessage, FileContent, ImageContent, TextContent, ToolCall
from haystack.token_counters import OpenAITokenCounter
from haystack.token_counters import openai_counter as openai_counter_module
from haystack.tools import tool
from haystack.utils import Secret


@tool
def search(query: Annotated[str, "the search query"]) -> str:
    """Search the web for a query and return the top results."""
    return "result"


@tool
def inspect_image(question: Annotated[str, "question about the image"]) -> str:
    """Inspect an image."""
    return "image inspected"


@tool
def inspect_file(question: Annotated[str, "question about the file"]) -> str:
    """Inspect a file."""
    return "file inspected"


class TestOpenAITokenCounter:
    def test_empty_input_does_not_initialize_the_client(self):
        counter = OpenAITokenCounter("gpt-5-mini")
        assert counter.count([]) == 0
        assert counter.client is None

    def test_warm_up_initializes_the_client_once(self, monkeypatch):
        openai_class = MagicMock()
        monkeypatch.setattr(openai_counter_module, "OpenAI", openai_class)
        monkeypatch.setenv("OPENAI_TIMEOUT", "12.5")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "3")
        counter = OpenAITokenCounter(
            "gpt-5-mini",
            api_key=Secret.from_token("test-key"),
            api_base_url="https://example.com/v1",
            organization="org-test",
        )
        counter.warm_up()
        counter.warm_up()
        openai_class.assert_called_once_with(
            api_key="test-key",
            organization="org-test",
            base_url="https://example.com/v1",
            timeout=12.5,
            max_retries=3,
            http_client=None,
        )

    def test_count_calls_the_input_token_endpoint(self):
        counter = OpenAITokenCounter("gpt-5-mini", api_key=Secret.from_token("test-key"))
        client = MagicMock()
        client.responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=42)
        counter.client = client
        image = ImageContent(base64_image="Zm9v", mime_type="image/png")
        messages = [ChatMessage.from_system("Be concise."), ChatMessage.from_user(content_parts=["Look:", image])]

        result = counter.count(messages, tools=[search])
        assert result == 42
        client.responses.input_tokens.count.assert_called_once_with(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": "Be concise."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Look:"},
                        {"type": "input_image", "image_url": "data:image/png;base64,Zm9v"},
                    ],
                },
            ],
            tools=[{"type": "function", **search.tool_spec}],
        )

    def test_close_releases_the_client(self):
        counter = OpenAITokenCounter("gpt-5-mini", api_key=Secret.from_token("test-key"))
        client = MagicMock()
        counter.client = client
        counter.close()
        client.close.assert_called_once_with()
        assert counter.client is None

    def test_serde_round_trip(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "test-key")
        data = OpenAITokenCounter(
            "gpt-5-mini",
            api_key=Secret.from_env_var("CUSTOM_OPENAI_API_KEY"),
            api_base_url="https://example.com/v1",
            organization="org-test",
            timeout=20.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://example.com:8080"},
        ).to_dict()

        assert data == {
            "type": "haystack.token_counters.openai_counter.OpenAITokenCounter",
            "init_parameters": {
                "model": "gpt-5-mini",
                "api_key": {"type": "env_var", "env_vars": ["CUSTOM_OPENAI_API_KEY"], "strict": True},
                "api_base_url": "https://example.com/v1",
                "organization": "org-test",
                "timeout": 20.0,
                "max_retries": 2,
                "http_client_kwargs": {"proxy": "http://example.com:8080"},
            },
        }
        restored = OpenAITokenCounter.from_dict(data)
        assert isinstance(restored, OpenAITokenCounter)
        assert restored.model == "gpt-5-mini"
        assert restored.api_key == Secret.from_env_var("CUSTOM_OPENAI_API_KEY")
        assert restored.api_base_url == "https://example.com/v1"
        assert restored.http_client_kwargs == {"proxy": "http://example.com:8080"}


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.integration
class TestOpenAITokenCounterIntegration:
    def test_count_complex(self, base64_image_string, base64_pdf_string):
        """Add messages with multiple content types and tool calls to test the token counting."""
        image = ImageContent(base64_image=base64_image_string, mime_type="image/png")
        file = FileContent(base64_data=base64_pdf_string, mime_type="application/pdf", filename="sample.pdf")
        image_call = ToolCall(
            tool_name="inspect_image",
            arguments={"question": "What is in this image?"},
            id="fc_test_image",
            extra={"call_id": "call_test_image"},
        )
        file_call = ToolCall(
            tool_name="inspect_file",
            arguments={"question": "Summarize this file."},
            id="fc_test_file",
            extra={"call_id": "call_test_file"},
        )
        messages = [
            ChatMessage.from_system("You inspect supplied images and files."),
            ChatMessage.from_user(content_parts=["Inspect both attachments.", image, file]),
            ChatMessage.from_assistant("I will inspect both attachments.", tool_calls=[image_call, file_call]),
            ChatMessage.from_tool(tool_result=[TextContent(text="Image inspection result:"), image], origin=image_call),
            ChatMessage.from_tool(tool_result=[TextContent(text="File inspection result:"), file], origin=file_call),
            ChatMessage.from_assistant("The image and file have both been inspected."),
        ]
        counter = OpenAITokenCounter("gpt-5-mini-2025-08-07")
        assert counter.count(messages, tools=[inspect_image, inspect_file]) == 300
