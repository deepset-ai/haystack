# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import ValidationError, validate


def is_valid_json(s: str) -> bool:
    """
    Check if the provided string is a valid JSON.

    :param s: The string to be checked.
    :returns: `True` if the string is a valid JSON; otherwise, `False`.
    """
    try:
        json.loads(s)
    except ValueError:
        return False
    return True


@component
class JsonSchemaValidator:
    """
    Validates JSON content of `ChatMessage` against a specified [JSON Schema](https://json-schema.org/).

    If JSON content of a message conforms to the provided schema, the message is passed along the "validated" output.
    If the JSON content does not conform to the schema, the message is passed along the "validation_error" output.
    In the latter case, the error message is constructed using the provided `error_template` or a default template.
    These error ChatMessages can be used by LLMs in Haystack 2.x recovery loops.

    Usage example:

    ```python
    from typing import List

    from haystack import Pipeline
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.components.joiners import BranchJoiner
    from haystack.components.validators import JsonSchemaValidator
    from haystack import component
    from haystack.dataclasses import ChatMessage


    @component
    class MessageProducer:

        @component.output_types(messages=List[ChatMessage])
        def run(self, messages: List[ChatMessage]) -> dict:
            return {"messages": messages}


    p = Pipeline()
    p.add_component("llm", OpenAIChatGenerator(model="gpt-4-1106-preview",
                                               generation_kwargs={"response_format": {"type": "json_object"}}))
    p.add_component("schema_validator", JsonSchemaValidator())
    p.add_component("joiner_for_llm", BranchJoiner(List[ChatMessage]))
    p.add_component("message_producer", MessageProducer())

    p.connect("message_producer.messages", "joiner_for_llm")
    p.connect("joiner_for_llm", "llm")
    p.connect("llm.replies", "schema_validator.messages")
    p.connect("schema_validator.validation_error", "joiner_for_llm")

    result = p.run(data={
        "message_producer": {
            "messages":[ChatMessage.from_user("Generate JSON for person with name 'John' and age 30")]},
            "schema_validator": {
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
    })
    print(result)
    >> {'schema_validator': {'validated': [ChatMessage(content='\\n{\\n  "name": "John",\\n  "age": 30\\n}',
    role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-4-1106-preview', 'index': 0,
    'finish_reason': 'stop', 'usage': {'completion_tokens': 17, 'prompt_tokens': 20, 'total_tokens': 37}})]}}
    ```
    """

    # Default error description template
    default_error_template = (
        "The following generated JSON does not conform to the provided schema.\n"
        "Generated JSON: {failing_json}\n"
        "Error details:\n- Message: {error_message}\n"
        "- Error Path in JSON: {error_path}\n"
        "- Schema Path: {error_schema_path}\n"
        "Please match the following schema:\n"
        "{json_schema}\n"
        "and provide the corrected JSON content ONLY. Please do not output anything else than the raw corrected "
        "JSON string, this is the most important part of the task. Don't use any markdown and don't add any comment."
    )

    def __init__(self, json_schema: Optional[Dict[str, Any]] = None, error_template: Optional[str] = None):
        """
        Initialize the JsonSchemaValidator component.

        :param json_schema: A dictionary representing the [JSON schema](https://json-schema.org/) against which
            the messages' content is validated.
        :param error_template: A custom template string for formatting the error message in case of validation failure.
        """
        jsonschema_import.check()
        self.json_schema = json_schema
        self.error_template = error_template

    @component.output_types(validated=List[ChatMessage], validation_error=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        json_schema: Optional[Dict[str, Any]] = None,
        error_template: Optional[str] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Validates the last of the provided messages against the specified json schema.

        If it does, the message is passed along the "validated" output. If it does not, the message is passed along
        the "validation_error" output.

        :param messages: A list of ChatMessage instances to be validated. The last message in this list is the one
            that is validated.
        :param json_schema: A dictionary representing the [JSON schema](https://json-schema.org/)
            against which the messages' content is validated. If not provided, the schema from the component init
            is used.
        :param error_template: A custom template string for formatting the error message in case of validation. If not
            provided, the `error_template` from the component init is used.
        :return:  A dictionary with the following keys:
            - "validated": A list of messages if the last message is valid.
            - "validation_error": A list of messages if the last message is invalid.
        :raises ValueError: If no JSON schema is provided or if the message content is not a dictionary or a list of
            dictionaries.
        """
        last_message = messages[-1]
        if last_message.text is None:
            raise ValueError(f"The provided ChatMessage has no text. ChatMessage: {last_message}")
        if not is_valid_json(last_message.text):
            return {
                "validation_error": [
                    ChatMessage.from_user(
                        f"The message '{last_message.text}' is not a valid JSON object. "
                        f"Please provide only a valid JSON object in string format."
                        f"Don't use any markdown and don't add any comment."
                    )
                ]
            }

        last_message_content = json.loads(last_message.text)
        json_schema = json_schema or self.json_schema
        error_template = error_template or self.error_template or self.default_error_template

        if not json_schema:
            raise ValueError("Provide a JSON schema for validation either in the run method or in the component init.")
        # fc payload is json object but subtree `parameters` is string - we need to convert to json object
        # we need complete json to validate it against schema
        last_message_json = self._recursive_json_to_object(last_message_content)
        using_openai_schema: bool = self._is_openai_function_calling_schema(json_schema)
        if using_openai_schema:
            validation_schema = json_schema["parameters"]
        else:
            validation_schema = json_schema
        try:
            last_message_json = [last_message_json] if not isinstance(last_message_json, list) else last_message_json
            for content in last_message_json:
                if using_openai_schema:
                    validate(instance=content["function"]["arguments"], schema=validation_schema)
                else:
                    validate(instance=content, schema=validation_schema)

            return {"validated": [last_message]}
        except ValidationError as e:
            error_path = " -> ".join(map(str, e.absolute_path)) if e.absolute_path else "N/A"
            error_schema_path = " -> ".join(map(str, e.absolute_schema_path)) if e.absolute_schema_path else "N/A"

            error_template = error_template or self.default_error_template

            recovery_prompt = self._construct_error_recovery_message(
                error_template, str(e), error_path, error_schema_path, validation_schema, failing_json=last_message.text
            )
            return {"validation_error": [ChatMessage.from_user(recovery_prompt)]}

    def _construct_error_recovery_message(  # pylint: disable=too-many-positional-arguments
        self,
        error_template: str,
        error_message: str,
        error_path: str,
        error_schema_path: str,
        json_schema: Dict[str, Any],
        failing_json: str,
    ) -> str:
        """
        Constructs an error recovery message using a specified template or the default one if none is provided.

        :param error_template: A custom template string for formatting the error message in case of validation failure.
        :param error_message: The error message returned by the JSON schema validator.
        :param error_path: The path in the JSON content where the error occurred.
        :param error_schema_path: The path in the JSON schema where the error occurred.
        :param json_schema: The JSON schema against which the content is validated.
        :param failing_json: The generated invalid JSON string.
        """
        error_template = error_template or self.default_error_template

        return error_template.format(
            error_message=error_message,
            error_path=error_path,
            error_schema_path=error_schema_path,
            json_schema=json_schema,
            failing_json=failing_json,
        )

    def _is_openai_function_calling_schema(self, json_schema: Dict[str, Any]) -> bool:
        """
        Checks if the provided schema is a valid OpenAI function calling schema.

        :param json_schema: The JSON schema to check
        :return: `True` if the schema is a valid OpenAI function calling schema; otherwise, `False`.
        """
        return all(key in json_schema for key in ["name", "description", "parameters"])

    def _recursive_json_to_object(self, data: Any) -> Any:
        """
        Convert any string values that are valid JSON objects into dictionary objects.

        Returns a new data structure.

        :param data: The data structure to be traversed.
        :return: A new data structure with JSON strings converted to dictionary objects.
        """
        if isinstance(data, list):
            return [self._recursive_json_to_object(item) for item in data]

        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        json_value = json.loads(value)
                        if isinstance(json_value, (dict, list)):
                            new_dict[key] = self._recursive_json_to_object(json_value)
                        else:
                            new_dict[key] = value  # Preserve the original string value
                    except json.JSONDecodeError:
                        new_dict[key] = value
                elif isinstance(value, dict):
                    new_dict[key] = self._recursive_json_to_object(value)
                else:
                    new_dict[key] = value
            return new_dict

        # If it's neither a list nor a dictionary, return the value directly
        raise ValueError("Input must be a dictionary or a list of dictionaries.")
