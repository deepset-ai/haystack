# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
from typing import List

import pytest

from haystack import Pipeline, component
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage


@pytest.fixture
def genuine_fc_message():
    return """[{"id": "call_NJr1NBz2Th7iUWJpRIJZoJIA", "function": {"arguments": "{\\n    \\"basehead\\": \\"main...amzn_chat\\",\\n    \\"owner\\": \\"deepset-ai\\",\\n    \\"repo\\": \\"haystack-core-integrations\\"\\n  }", "name": "compare_branches"}, "type": "function"}]"""


@pytest.fixture
def json_schema_github_compare():
    json_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "A unique identifier for the call"},
            "function": {
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "basehead": {
                                "type": "string",
                                "pattern": "^[^\\.]+(\\.{3}).+$",
                                "description": "Branch names must be in the format 'base_branch...head_branch'",
                            },
                            "owner": {"type": "string", "description": "Owner of the repository"},
                            "repo": {"type": "string", "description": "Name of the repository"},
                        },
                        "required": ["basehead", "owner", "repo"],
                        "description": "Parameters for the function call",
                    },
                    "name": {"type": "string", "description": "Name of the function to be called"},
                },
                "required": ["arguments", "name"],
                "description": "Details of the function being called",
            },
            "type": {"type": "string", "description": "Type of the call (e.g., 'function')"},
        },
        "required": ["function", "type"],
        "description": "Structure representing a function call",
    }
    return json_schema


@pytest.fixture
def json_schema_github_compare_openai():
    json_schema = {
        "name": "compare_branches",
        "description": "Compares two branches in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "basehead": {
                    "type": "string",
                    "pattern": "^[^\\.]+(\\.{3}).+$",
                    "description": "Branch names must be in the format 'base_branch...head_branch'",
                },
                "owner": {"type": "string", "description": "Owner of the repository"},
                "repo": {"type": "string", "description": "Name of the repository"},
            },
            "required": ["basehead", "owner", "repo"],
            "description": "Parameters for the function call",
        },
    }
    return json_schema


class TestJsonSchemaValidator:
    #  Validates a message against a provided JSON schema successfully.
    def test_validates_message_against_json_schema(self, json_schema_github_compare, genuine_fc_message):
        validator = JsonSchemaValidator()
        message = ChatMessage.from_assistant(genuine_fc_message)

        result = validator.run([message], json_schema_github_compare)

        assert "validated" in result
        assert len(result["validated"]) == 1
        assert result["validated"][0] == message

    # Validates recursive_json_to_object method
    def test_recursive_json_to_object(self, genuine_fc_message):
        arguments_is_string = json.loads(genuine_fc_message)
        assert isinstance(arguments_is_string[0]["function"]["arguments"], str)

        # but ensure_json_objects converts the string to a json object
        validator = JsonSchemaValidator()
        result = validator._recursive_json_to_object({"key": genuine_fc_message})

        # we need this recursive json conversion to validate the message
        assert result["key"][0]["function"]["arguments"]["basehead"] == "main...amzn_chat"

    #  Validates multiple messages against a provided JSON schema successfully.
    def test_validates_multiple_messages_against_json_schema(self, json_schema_github_compare, genuine_fc_message):
        validator = JsonSchemaValidator()

        messages = [
            ChatMessage.from_user("I'm not being validated, but the message after me is!"),
            ChatMessage.from_assistant(genuine_fc_message),
        ]

        result = validator.run(messages, json_schema_github_compare)
        assert "validated" in result
        assert len(result["validated"]) == 1
        assert result["validated"][0] == messages[1]

    #  Validates a message against an OpenAI function calling schema successfully.
    def test_validates_message_against_openai_function_calling_schema(
        self, json_schema_github_compare_openai, genuine_fc_message
    ):
        validator = JsonSchemaValidator()

        message = ChatMessage.from_assistant(genuine_fc_message)
        result = validator.run([message], json_schema_github_compare_openai)

        assert "validated" in result
        assert len(result["validated"]) == 1
        assert result["validated"][0] == message

    #  Validates multiple messages against an OpenAI function calling schema successfully.
    def test_validates_multiple_messages_against_openai_function_calling_schema(
        self, json_schema_github_compare_openai, genuine_fc_message
    ):
        validator = JsonSchemaValidator()

        messages = [
            ChatMessage.from_system("Common use case is that this is for example system message"),
            ChatMessage.from_assistant(genuine_fc_message),
        ]

        result = validator.run(messages, json_schema_github_compare_openai)

        assert "validated" in result
        assert len(result["validated"]) == 1
        assert result["validated"][0] == messages[1]

    #  Constructs a custom error recovery message when validation fails.
    def test_construct_custom_error_recovery_message(self):
        validator = JsonSchemaValidator()

        new_error_template = (
            "Error details:\n- Message: {error_message}\n"
            "- Error Path in JSON: {error_path}\n"
            "- Schema Path: {error_schema_path}\n"
            "Please match the following schema:\n"
            "{json_schema}\n"
            "Failing Json: {failing_json}\n"
        )

        recovery_message = validator._construct_error_recovery_message(
            new_error_template, "Error message", "Error path", "Error schema path", {"type": "object"}, "Failing Json"
        )

        expected_recovery_message = (
            "Error details:\n- Message: Error message\n"
            "- Error Path in JSON: Error path\n"
            "- Schema Path: Error schema path\n"
            "Please match the following schema:\n"
            "{'type': 'object'}\n"
            "Failing Json: Failing Json\n"
        )
        assert recovery_message == expected_recovery_message

    def test_schema_validator_in_pipeline_validated(self, json_schema_github_compare, genuine_fc_message):
        @component
        class ChatMessageProducer:
            @component.output_types(messages=List[ChatMessage])
            def run(self):
                return {"messages": [ChatMessage.from_assistant(genuine_fc_message)]}

        pipe = Pipeline()
        pipe.add_component(name="schema_validator", instance=JsonSchemaValidator())
        pipe.add_component(name="message_producer", instance=ChatMessageProducer())
        pipe.connect("message_producer", "schema_validator")
        result = pipe.run(data={"schema_validator": {"json_schema": json_schema_github_compare}})
        assert "validated" in result["schema_validator"]
        assert len(result["schema_validator"]["validated"]) == 1
        assert result["schema_validator"]["validated"][0].text == genuine_fc_message

    def test_schema_validator_in_pipeline_validation_error(self, json_schema_github_compare):
        @component
        class ChatMessageProducer:
            @component.output_types(messages=List[ChatMessage])
            def run(self):
                # example json string that is not valid
                simple_invalid_json = '{"key": "value"}'
                return {"messages": [ChatMessage.from_assistant(simple_invalid_json)]}  # invalid message

        pipe = Pipeline()
        pipe.add_component(name="schema_validator", instance=JsonSchemaValidator())
        pipe.add_component(name="message_producer", instance=ChatMessageProducer())
        pipe.connect("message_producer", "schema_validator")
        result = pipe.run(data={"schema_validator": {"json_schema": json_schema_github_compare}})
        assert "validation_error" in result["schema_validator"]
        assert len(result["schema_validator"]["validation_error"]) == 1
        assert "Error details" in result["schema_validator"]["validation_error"][0].text
