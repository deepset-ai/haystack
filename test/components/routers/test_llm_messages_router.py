# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import Any
from unittest.mock import Mock

import pytest

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.routers.llm_messages_router import LLMMessagesRouter
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage


class MockChatGenerator:
    def __init__(self, return_text: str = "safe"):
        self.return_text = return_text

    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        return {"replies": [ChatMessage.from_assistant(self.return_text)]}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, return_text=self.return_text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MockChatGenerator":
        return default_from_dict(cls, data)


class TestLLMMessagesRouter:
    def test_init(self):
        system_prompt = "Classify the messages as safe or unsafe."
        chat_generator = MockChatGenerator()

        router = LLMMessagesRouter(
            chat_generator=chat_generator,
            system_prompt=system_prompt,
            output_names=["safe", "unsafe"],
            output_patterns=["safe", "unsafe"],
        )

        assert router._chat_generator is chat_generator
        assert router._system_prompt == system_prompt
        assert router._output_names == ["safe", "unsafe"]
        assert router._output_patterns == ["safe", "unsafe"]
        assert router._compiled_patterns == [re.compile(pattern) for pattern in ["safe", "unsafe"]]
        assert router._is_warmed_up is False

    def test_init_errors(self):
        chat_generator = MockChatGenerator()

        with pytest.raises(ValueError):
            LLMMessagesRouter(chat_generator=chat_generator, output_names=[], output_patterns=["pattern1", "pattern2"])

        with pytest.raises(ValueError):
            LLMMessagesRouter(chat_generator=chat_generator, output_names=["name1", "name2"], output_patterns=[])

        with pytest.raises(ValueError):
            LLMMessagesRouter(
                chat_generator=chat_generator, output_names=["name1", "name2"], output_patterns=["pattern1"]
            )

    def test_warm_up_with_unwarmable_chat_generator(self):
        chat_generator = MockChatGenerator()
        router = LLMMessagesRouter(
            chat_generator=chat_generator, output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )
        router.warm_up()
        assert router._is_warmed_up is True

    def test_warm_up_with_warmable_chat_generator(self):
        chat_generator = Mock()
        router = LLMMessagesRouter(
            chat_generator=chat_generator, output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )
        router.warm_up()
        assert router._is_warmed_up is True
        assert router._chat_generator.warm_up.call_count == 1

    def test_run_input_errors(self):
        router = LLMMessagesRouter(
            chat_generator=MockChatGenerator(), output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )

        with pytest.raises(ValueError):
            router.run([])

        with pytest.raises(ValueError):
            router.run([ChatMessage.from_system("You are a helpful assistant.")])

    def test_run_no_warm_up_with_unwarmable_chat_generator(self):
        router = LLMMessagesRouter(
            chat_generator=MockChatGenerator(), output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )

        router.run([ChatMessage.from_user("Hello")])

    def test_run_no_warm_up_with_warmable_chat_generator(self):
        chat_generator = Mock()
        router = LLMMessagesRouter(
            chat_generator=chat_generator, output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )

        with pytest.raises(RuntimeError):
            router.run([ChatMessage.from_user("Hello")])

    def test_run(self):
        router = LLMMessagesRouter(
            chat_generator=MockChatGenerator(return_text="safe"),
            output_names=["safe", "unsafe"],
            output_patterns=["safe", "unsafe"],
        )

        messages = [ChatMessage.from_user("Hello")]
        result = router.run(messages)

        assert result["chat_generator_text"] == "safe"
        assert result["safe"] == messages
        assert "unsafe" not in result
        assert "unmatched" not in result

    def test_run_with_system_prompt(self):
        chat_generator = Mock()
        chat_generator.run.return_value = {"replies": [ChatMessage.from_assistant("safe")]}

        system_prompt = "Classify the messages as safe or unsafe."

        router = LLMMessagesRouter(
            chat_generator=chat_generator,
            output_names=["safe", "unsafe"],
            output_patterns=["safe", "unsafe"],
            system_prompt=system_prompt,
        )
        router.warm_up()

        messages = [ChatMessage.from_user("Hello")]
        router.run(messages)

        chat_generator.run.assert_called_once_with(messages=[ChatMessage.from_system(system_prompt)] + messages)

    def test_run_unmatched_output(self):
        router = LLMMessagesRouter(
            chat_generator=MockChatGenerator(return_text="irrelevant"),
            output_names=["safe", "unsafe"],
            output_patterns=["safe", "unsafe"],
        )

        messages = [ChatMessage.from_user("Hello")]
        result = router.run(messages)

        assert result["chat_generator_text"] == "irrelevant"
        assert result["unmatched"] == messages
        assert "safe" not in result
        assert "unsafe" not in result

    def test_to_dict(self):
        chat_generator = MockChatGenerator(return_text="safe")

        router = LLMMessagesRouter(
            chat_generator=chat_generator, output_names=["safe", "unsafe"], output_patterns=["safe", "unsafe"]
        )

        result = router.to_dict()

        assert result["type"] == "haystack.components.routers.llm_messages_router.LLMMessagesRouter"
        assert result["init_parameters"]["chat_generator"] == chat_generator.to_dict()
        assert result["init_parameters"]["output_names"] == ["safe", "unsafe"]
        assert result["init_parameters"]["output_patterns"] == ["safe", "unsafe"]
        assert result["init_parameters"]["system_prompt"] is None

    def test_from_dict(self):
        chat_generator = MockChatGenerator(return_text="safe")

        data = {
            "type": "haystack.components.routers.llm_messages_router.LLMMessagesRouter",
            "init_parameters": {
                "chat_generator": chat_generator.to_dict(),
                "output_names": ["safe", "unsafe"],
                "output_patterns": ["safe", "unsafe"],
                "system_prompt": None,
            },
        }

        router = LLMMessagesRouter.from_dict(data)

        assert router._chat_generator.to_dict() == chat_generator.to_dict()
        assert router._output_names == ["safe", "unsafe"]
        assert router._output_patterns == ["safe", "unsafe"]
        assert router._system_prompt is None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run(self):
        system_prompt = "Classify the messages into safe or unsafe. Respond with the label only, no other text."
        router = LLMMessagesRouter(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
            system_prompt=system_prompt,
            output_names=["safe", "unsafe"],
            output_patterns=[r"(?i)safe", r"(?i)unsafe"],
        )

        messages = [ChatMessage.from_user("Hello")]
        router.warm_up()
        result = router.run(messages)
        print(result)

        assert result["safe"] == messages
        assert result["chat_generator_text"].lower() == "safe"
        assert "unsafe" not in result
        assert "unmatched" not in result
