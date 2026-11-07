# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack import Pipeline, component
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage


@pytest.fixture
def genuine_fc_message():
    return """[{"id": "call_NJr1NBz2Th7iUWJpRIJZoJIA", "function": {"arguments": "{\\n    \\"basehead\\": \\"main...amzn_chat\\",\\n    \\"owner\\": \\"deepset-ai\\",\\n    \\"repo\\": \\"haystack-core-integrations\\"\\n  }", "name": "compare_branches"}, "type": "function"}]"""  # noqa: E501


@pytest.fixture
def json_schema_github_compare():
    return {
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


@pytest.fixture
def json_schema_github_compare_openai():
    return {
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


class TestJsonSchemaValidator:
    #  Validates a message against a provided JSON schema successfully.
    def test_validates_message_against_json_schema(self, json_schema_github_compare, genuine_fc_message):
        validator = JsonSchemaValidator()
        message = ChatMessage.from_assistant(genuine_fc_message)

        result = validator.run([message], json_schema_github_compare)

        assert "validated" in result
        assert len(result["validated"]) == 1
        assert result["validated"][0] == message

    def test_run_raises_clear_error_for_empty_messages(self, json_schema_github_compare):
        validator = JsonSchemaValidator(json_schema=json_schema_github_compare)

        with pytest.raises(ValueError, match="The provided list of messages is empty"):
            validator.run([])

    def test_accepts_empty_json_schema(self):
        validator = JsonSchemaValidator(json_schema={})
        message = ChatMessage.from_assistant('{"anything": "is valid"}')

        result = validator.run([message])

        assert result == {"validated": [message]}

    def test_run_empty_json_schema_overrides_init_schema(self):
        validator = JsonSchemaValidator(json_schema={"type": "object", "required": ["name"]})
        message = ChatMessage.from_assistant("{}")

        result = validator.run([message], json_schema={})

        assert result == {"validated": [message]}

    @pytest.mark.parametrize("schema_type", ["array", ["array", "null"]])
    @pytest.mark.parametrize(
        "constraints, content, error_schema_path",
        [
            ({"items": {"type": "integer"}}, "[1, 2]", None),
            ({"items": {"type": "integer"}}, '[1, "two"]', "items -> type"),
            ({"minItems": 1}, "[]", "minItems"),
            ({"maxItems": 1}, "[1, 2]", "maxItems"),
            ({"uniqueItems": True}, "[1, 1]", "uniqueItems"),
        ],
    )
    def test_validates_top_level_array_as_a_whole(self, schema_type, constraints, content, error_schema_path):
        validator = JsonSchemaValidator(json_schema={"type": schema_type, **constraints})
        message = ChatMessage.from_assistant(content)

        result = validator.run([message])

        if error_schema_path is None:
            assert result == {"validated": [message]}
        else:
            assert list(result) == ["validation_error"]
            assert len(result["validation_error"]) == 1
            recovery_text = result["validation_error"][0].text
            assert recovery_text is not None
            assert f"Schema Path: {error_schema_path}" in recovery_text

    @pytest.mark.parametrize(
        "schema_type, content, output", [(["array", "null"], "null", "validated"), ("array", "42", "validation_error")]
    )
    def test_array_schema_validates_non_array_content(self, schema_type, content, output):
        validator = JsonSchemaValidator(json_schema={"type": schema_type})

        result = validator.run([ChatMessage.from_assistant(content)])

        assert list(result) == [output]

    # Validates recursive_json_to_object method
    def test_recursive_json_to_object(self, genuine_fc_message):
        arguments_is_string = json.loads(genuine_fc_message)
        assert isinstance(arguments_is_string[0]["function"]["arguments"], str)

        # but ensure_json_objects converts the string to a json object
        validator = JsonSchemaValidator()
        result = validator._recursive_json_to_object({"key": genuine_fc_message})

        # we need this recursive json conversion to validate the message
        assert result["key"][0]["function"]["arguments"]["basehead"] == "main...amzn_chat"

    @pytest.mark.parametrize("scalar", ["hello", 42, 3.14, True, None])
    def test_recursive_json_to_object_with_top_level_scalar(self, scalar):
        validator = JsonSchemaValidator()

        assert validator._recursive_json_to_object(scalar) == scalar

    def test_recursive_json_to_object_with_list_of_scalars(self):
        validator = JsonSchemaValidator()

        assert validator._recursive_json_to_object([1, "two", None]) == [1, "two", None]

    def test_validates_message_with_top_level_json_scalar(self):
        validator = JsonSchemaValidator(json_schema={"type": "string"})

        result = validator.run([ChatMessage.from_assistant('"hello"')])

        assert "validated" in result
        assert len(result["validated"]) == 1

    @pytest.mark.parametrize("message_text", ["42", "true", "null"])
    def test_validation_error_for_top_level_json_scalar(self, message_text):
        validator = JsonSchemaValidator(json_schema={"type": "string"})

        result = validator.run([ChatMessage.from_assistant(message_text)])

        assert "validation_error" in result
        assert len(result["validation_error"]) == 1

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

    def test_validates_function_call_list_with_array_arguments(self):
        validator = JsonSchemaValidator(
            json_schema={
                "name": "collect",
                "description": "Collect integers.",
                "parameters": {"type": "array", "items": {"type": "integer"}},
            }
        )
        message = ChatMessage.from_assistant(
            json.dumps([{"function": {"arguments": "[1, 2]"}}, {"function": {"arguments": "[3, 4]"}}])
        )

        result = validator.run([message])

        assert result == {"validated": [message]}

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
            @component.output_types(messages=list[ChatMessage])
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
            @component.output_types(messages=list[ChatMessage])
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
