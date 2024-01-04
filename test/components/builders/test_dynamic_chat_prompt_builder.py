from typing import List

import pytest

from haystack.components.builders import DynamicChatPromptBuilder
from haystack.dataclasses import ChatMessage


class TestDynamicChatPromptBuilder:
    def test_initialization(self):
        runtime_variables = ["var1", "var2", "var3"]
        builder = DynamicChatPromptBuilder(runtime_variables)
        assert builder.runtime_variables == runtime_variables

        # we have inputs that contain: prompt_source, template_variables + runtime_variables
        expected_keys = set(runtime_variables + ["prompt_source", "template_variables"])
        assert set(builder.__canals_input__.keys()) == expected_keys

        # response is always prompt regardless of chat mode
        assert set(builder.__canals_output__.keys()) == {"prompt"}

        # prompt_source is a list of ChatMessage
        assert builder.__canals_input__["prompt_source"].type == List[ChatMessage]

        # output is always prompt, but the type is different depending on the chat mode
        assert builder.__canals_output__["prompt"].type == List[ChatMessage]

    def test_non_empty_chat_messages(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])
        prompt_source = [ChatMessage.from_system(content="Hello"), ChatMessage.from_user(content="Hello, {{ who }}!")]
        template_variables = {"who": "World"}

        result = prompt_builder.run(prompt_source, template_variables)

        assert result == {
            "prompt": [ChatMessage.from_system(content="Hello"), ChatMessage.from_user(content="Hello, World!")]
        }

    def test_single_chat_message(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])
        prompt_source = [ChatMessage.from_user(content="Hello, {{ who }}!")]
        template_variables = {"who": "World"}

        result = prompt_builder._process_chat_messages(prompt_source, template_variables)

        assert result == [ChatMessage.from_user(content="Hello, World!")]

    def test_empty_chat_message_list(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])

        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(prompt_source=[], template_variables={})

    def test_chat_message_list_with_mixed_object_list(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])

        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(
                prompt_source=[ChatMessage.from_user("Hello"), "there world"], template_variables={}
            )

    def test_chat_message_list_with_missing_variables(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])
        prompt_source = [ChatMessage.from_user(content="Hello, {{ who }}!")]

        # Call the _process_chat_messages method and expect a ValueError
        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(prompt_source, template_variables={})

    def test_missing_template_variables(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])

        # missing template variable city
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name"})

        # missing template variable name
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"city"})

        # completely unknown template variable
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"age"})

    def test_provided_template_variables(self):
        prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])

        # both variables are provided
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city", "age"})
