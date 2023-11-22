from typing import List, Union

import pytest
from jinja2 import TemplateSyntaxError

from haystack.preview.components.builders.dynamic_prompt_builder import DynamicPromptBuilder
from haystack.preview.dataclasses import ChatMessage


class TestDynamicPromptBuilder:
    def test_initialization_chat_on(self):
        expected_runtime_variables = ["var1", "var2", "var3"]
        builder = DynamicPromptBuilder(expected_runtime_variables, chat_mode=True)
        assert builder.expected_runtime_variables == expected_runtime_variables
        assert builder.chat_mode

        # regardless of the chat mode
        # we have inputs that contain: prompt_source, template_variables + expected_runtime_variables
        expected_keys = set(expected_runtime_variables + ["prompt_source", "template_variables"])
        assert set(builder.__canals_input__.keys()) == expected_keys

        # response is always prompt regardless of chat mode
        assert set(builder.__canals_output__.keys()) == {"prompt"}

        # prompt_source is a list of ChatMessage or a string
        assert builder.__canals_input__["prompt_source"].type == Union[List[ChatMessage], str]

        # output is always prompt, but the type is different depending on the chat mode
        assert builder.__canals_output__["prompt"].type == List[ChatMessage]

    def test_initialization_chat_off(self):
        expected_runtime_variables = ["var1", "var2"]
        builder = DynamicPromptBuilder(expected_runtime_variables, False)
        assert builder.expected_runtime_variables == expected_runtime_variables
        assert not builder.chat_mode

        # regardless of the chat mode
        # we have inputs that contain: prompt_source, template_variables + expected_runtime_variables
        expected_keys = set(expected_runtime_variables + ["prompt_source", "template_variables"])
        assert set(builder.__canals_input__.keys()) == expected_keys

        # response is always prompt regardless of chat mode
        assert set(builder.__canals_output__.keys()) == {"prompt"}

        # prompt_source is a list of ChatMessage or a string
        assert builder.__canals_input__["prompt_source"].type == Union[List[ChatMessage], str]

        # output is always prompt, but the type is different depending on the chat mode
        assert builder.__canals_output__["prompt"].type == str

    def test_to_dict_method_returns_expected_dictionary(self):
        expected_runtime_variables = ["var1", "var2", "var3"]
        chat_mode = True
        builder = DynamicPromptBuilder(expected_runtime_variables, chat_mode)
        expected_dict = {
            "type": "haystack.preview.components.builders.dynamic_prompt_builder.DynamicPromptBuilder",
            "init_parameters": {"expected_runtime_variables": expected_runtime_variables, "chat_mode": chat_mode},
        }
        assert builder.to_dict() == expected_dict

    def test_processing_a_simple_template_with_provided_variables(self):
        expected_runtime_variables = ["var1", "var2", "var3"]
        chat_mode = True

        builder = DynamicPromptBuilder(expected_runtime_variables, chat_mode)

        template = "Hello, {{ name }}!"
        template_variables = {"name": "John"}
        expected_result = "Hello, John!"

        assert builder._process_simple_template(template, template_variables) == expected_result

    def test_processing_a_simple_template_with_invalid_template(self):
        expected_runtime_variables = ["var1", "var2", "var3"]
        chat_mode = True

        builder = DynamicPromptBuilder(expected_runtime_variables, chat_mode)

        template = "Hello, {{ name }!"
        template_variables = {"name": "John"}
        with pytest.raises(TemplateSyntaxError):
            builder._process_simple_template(template, template_variables)

    def test_processing_a_simple_template_with_missing_variables(self):
        expected_runtime_variables = ["var1", "var2", "var3"]

        builder = DynamicPromptBuilder(expected_runtime_variables, False)

        with pytest.raises(ValueError):
            builder._process_simple_template("Hello, {{ name }}!", {})

    def test_non_empty_chat_messages(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"], chat_mode=True)
        prompt_source = [ChatMessage.from_system(content="Hello"), ChatMessage.from_user(content="Hello, {{ who }}!")]
        template_variables = {"who": "World"}

        result = prompt_builder._process_chat_messages(prompt_source, template_variables)

        assert result == [ChatMessage.from_system(content="Hello"), ChatMessage.from_user(content="Hello, World!")]

    def test_single_chat_message(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"], chat_mode=True)
        prompt_source = [ChatMessage.from_user(content="Hello, {{ who }}!")]
        template_variables = {"who": "World"}

        result = prompt_builder._process_chat_messages(prompt_source, template_variables)

        assert result == [ChatMessage.from_user(content="Hello, World!")]

    def test_empty_chat_message_list(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"], chat_mode=True)

        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(prompt_source=[], template_variables={})

    def test_chat_message_list_with_mixed_object_list(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"], chat_mode=True)

        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(
                prompt_source=[ChatMessage.from_user("Hello"), "there world"], template_variables={}
            )

    def test_chat_message_list_with_missing_variables(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"], chat_mode=True)
        prompt_source = [ChatMessage.from_user(content="Hello, {{ who }}!")]

        # Call the _process_chat_messages method and expect a ValueError
        with pytest.raises(ValueError):
            prompt_builder._process_chat_messages(prompt_source, template_variables={})

    def test_missing_template_variables(self):
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"])

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
        prompt_builder = DynamicPromptBuilder(expected_runtime_variables=["documents"])

        # both variables are provided
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city", "age"})
