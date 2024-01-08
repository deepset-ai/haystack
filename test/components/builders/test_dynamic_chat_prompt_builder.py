from typing import List

import pytest

from haystack import Pipeline
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

    def test_example_in_pipeline(self):
        # no parameter init, we don't use any runtime template variables
        prompt_builder = DynamicChatPromptBuilder()

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        location = "Berlin"
        system_message = ChatMessage.from_system(
            "You are a helpful assistant giving out valuable information to tourists."
        )
        messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

        res = pipe.run(
            data={"prompt_builder": {"template_variables": {"location": location}, "prompt_source": messages}}
        )
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists."),
                    ChatMessage.from_user("Tell me about Berlin"),
                ]
            }
        }

        messages = [
            system_message,
            ChatMessage.from_user("What's the weather forecast for {{location}} in the next {{day_count}} days?"),
        ]

        res = pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"location": location, "day_count": "5"},
                    "prompt_source": messages,
                }
            }
        )
        assert res == {
            "prompt_builder": {
                "prompt": [
                    ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists."),
                    ChatMessage.from_user("What's the weather forecast for Berlin in the next 5 days?"),
                ]
            }
        }
