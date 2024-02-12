import json
from typing import List, Any, Dict, Optional

from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import validate, ValidationError


@component
class JsonSchemaValidator:
    """
    JsonSchemaValidator validates JSON content of ChatMessage against a specified JSON schema.

    If JSON content of a message conforms to the provided schema, the message is passed along the "validated" output.
    If the JSON content does not conform to the schema, the message is passed along the "validation_error" output.
    In the latter case, the error message is constructed using the provided error_template or a default template.
    These error ChatMessages can be used by LLMs in Haystack 2.x recovery loops.
    """

    # Default error description template
    default_error_template = (
        "The JSON content in the previous message does not conform to the provided schema.\n"
        "Error details:\n- Message: {error_message}\n"
        "- Error Path in JSON: {error_path}\n"
        "- Schema Path: {error_schema_path}\n"
        "Please match the following schema:\n"
        "{json_schema}\n"
        "and provide the corrected JSON content ONLY."
    )

    def __init__(self):
        jsonschema_import.check()

    @component.output_types(validated=List[ChatMessage], validation_error=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        json_schema: Dict[str, Any],
        previous_messages: Optional[List[ChatMessage]] = None,
        error_template: Optional[str] = None,
    ):
        """
        Checks if the last message and its content field conforms to json_schema.

        :param messages: A list of ChatMessage instances to be validated. The last message in this list is the one
        that is validated.
        :param previous_messages: A list of previous ChatMessage instances, by default None. These are not validated
        but are returned in the case of an error.
        :param json_schema:A dictionary representing the JSON schema against which the messages' content is validated.
        :param error_template: A custom template string for formatting the error message in case of validation
        failure, by default None.
        """
        last_message = messages[-1]
        last_message_content = json.loads(last_message.content)

        # fc payload is json object but subtree `parameters` is string - we need to convert to json object
        # we need complete json to validate it against schema
        last_message_json = self.recursive_json_to_object(last_message_content)
        using_openai_schema: bool = self.is_openai_function_calling_schema(json_schema)
        if using_openai_schema:
            validation_schema = json_schema["parameters"]
        else:
            validation_schema = json_schema
        try:
            last_message_json = [last_message_json] if not isinstance(last_message_json, list) else last_message_json
            for content in last_message_json:
                if not self.is_function_calling_payload(content):
                    raise ValidationError(f"{content} is not a valid OpenAI function calling payload.")
                if using_openai_schema:
                    validate(instance=content["function"]["arguments"]["parameters"], schema=validation_schema)
                else:
                    validate(instance=content, schema=validation_schema)

            return {"validated": messages}
        except ValidationError as e:
            error_path = " -> ".join(map(str, e.absolute_path)) if e.absolute_path else "N/A"
            error_schema_path = " -> ".join(map(str, e.absolute_schema_path)) if e.absolute_schema_path else "N/A"

            error_template = error_template or self.default_error_template

            recovery_prompt = self.construct_error_recovery_message(
                error_template, str(e), error_path, error_schema_path, validation_schema
            )
            previous_messages = previous_messages or []
            complete_message_list = previous_messages + messages + [ChatMessage.from_user(recovery_prompt)]

            return {"validation_error": complete_message_list}

    def construct_error_recovery_message(
        self,
        error_template: str,
        error_message: str,
        error_path: str,
        error_schema_path: str,
        json_schema: Dict[str, Any],
    ) -> str:
        """
        Constructs an error recovery message using a specified template or the default one if none is provided.

        :param error_template: A custom template string for formatting the error message in case of validation failure.
        :param error_message: The error message returned by the JSON schema validator.
        :param error_path: The path in the JSON content where the error occurred.
        :param error_schema_path: The path in the JSON schema where the error occurred.
        :param json_schema: The JSON schema against which the content is validated.
        """
        error_template = error_template or self.default_error_template

        return error_template.format(
            error_message=error_message,
            error_path=error_path,
            error_schema_path=error_schema_path,
            json_schema=json_schema,
        )

    def is_openai_function_calling_schema(self, json_schema: Dict[str, Any]) -> bool:
        """
        Checks if the provided schema is a valid OpenAI function calling schema.

        :param json_schema: The JSON schema to check
        :return: True if the schema is a valid OpenAI function calling schema; otherwise, False.
        """
        return all(key in json_schema for key in ["name", "description", "parameters"])

    def is_function_calling_payload(self, content: Dict[str, Any]) -> bool:
        """
        Checks if the provided content is a valid OpenAI function calling payload.

        :param content: The content to check
        :return: True if the content is a valid OpenAI function calling payload; otherwise, False.
        """
        function = content.get("function")
        if isinstance(function, dict):
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                return "parameters" in arguments
        return False

    def recursive_json_to_object(self, data: Any) -> Any:
        """
        Recursively traverses a data structure (dictionary or list), converting any string values
        that are valid JSON objects into dictionary objects, and returns a new data structure.

        :param data: The data structure to be traversed.
        :return: A new data structure with JSON strings converted to dictionary objects.
        """
        if isinstance(data, list):
            return [self.recursive_json_to_object(item) for item in data]

        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        json_value = json.loads(value)
                        new_dict[key] = (
                            self.recursive_json_to_object(json_value)
                            if isinstance(json_value, (dict, list))
                            else json_value
                        )
                    except json.JSONDecodeError:
                        new_dict[key] = value
                elif isinstance(value, dict):
                    new_dict[key] = self.recursive_json_to_object(value)
                else:
                    new_dict[key] = value
            return new_dict

        # If it's neither a list nor a dictionary, return the value directly
        raise ValueError("Input must be a dictionary or a list of dictionaries.")
