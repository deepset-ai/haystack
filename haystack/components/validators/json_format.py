import json
from typing import Dict, List, Optional

from haystack import component
from haystack.dataclasses import ChatMessage

@component
class JsonFormatValidator:
    """
    Validates content of `ChatMessage` to JSON format.
    It allows to force LLM without json/tool mode to output valid JSON strings through a loop mechanism.

    If content of a message conforms to the JSON format, the message is passed along the "validated" output.
    If the content does not conform to the JSON format, the message is passed along the "validation_error" output.
    In the latter case, the error message is constructed using the provided `error_template` or a default template.
    These error ChatMessages can be used by LLMs in Haystack 2.x recovery loops.
    """

    # Default error description template
    default_error_template = (
        "The following generated JSON is not a valid JSON string.\n"
        "Generated JSON: {failing_json}\n"
        "Error details:\n- Message: {error_message}\n"
        "Please modify it to be a valid JSON string and output ONLY the corrected JSON content. Please do not output anything else than the raw JSON string, this is the most important part of the task. Don't use any markdown and don't add any comment."
    )

    def __init__(self, error_template: Optional[str] = None):
        """
        :param error_template: A custom template string for formatting the error message in case of validation failure.
        """
        self.error_template = error_template

    @component.output_types(
        validated=List[ChatMessage], validation_error=List[ChatMessage]
    )
    def run(
        self,
        messages: List[ChatMessage],
        error_template: Optional[str] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Validates the last of the provided messages to JSON format.

        If it does, the message is passed along the "validated" output. If it does not, the message is passed along
        the "validation_error" output.

        :param messages: A list of ChatMessage instances to be validated. The last message in this list is the one
            that is validated.
        :param error_template: A custom template string for formatting the error message in case of validation. If not
            provided, the `error_template` from the component init is used.
        :return:  A dictionary with the following keys:
            - "validated": A list of messages if the last message is valid.
            - "validation_error": A list of messages if the last message is invalid.
        :raises ValueError: If no JSON schema is provided or if the message content is not a dictionary or a list of
            dictionaries.
        """
        last_message = messages[-1]
        try:
            decoded_json_string = self._extract_json(last_message.content)
            if type(decoded_json_string) is str:
                messages.append(ChatMessage.from_user(decoded_json_string))
                return {"validated": [ChatMessage.from_assistant(decoded_json_string)]}
            return {"validated": [ChatMessage.from_assistant(json.dumps(decoded_json_string))]}
        except json.JSONDecodeError as e:
            error_template = (
                error_template or self.error_template or self.default_error_template
            )
            recovery_prompt = self._construct_error_recovery_message(
                error_template, str(e), failing_json=last_message.content
            )
            return {"validation_error": [ChatMessage.from_user(recovery_prompt)]}

    def _construct_error_recovery_message(
        self,
        error_template: str,
        error_message: str,
        failing_json: str,
    ) -> str:
        """
        Constructs an error recovery message using a specified template or the default one if none is provided.

        :param error_template: A custom template string for formatting the error message in case of validation failure.
        :param error_message: The error message returned by the JSON schema validator.
        :param failing_json: The generated invalid JSON string.
        """
        error_template = error_template or self.default_error_template

        return error_template.format(
            error_message=error_message,
            failing_json=failing_json,
        )

    def _extract_json(self, chat_content:str):
        start_index = chat_content.find('{')

        if start_index != -1:
            end_index = chat_content.rfind('}')

            if end_index != -1:
                json_string = chat_content[start_index:end_index + 1]
                json_data = json.loads(json_string)
                return json_data

            else:
                raise json.JSONDecodeError("No closing '}' found.", chat_content, end_index)
        else:
            raise json.JSONDecodeError("No opening '{' found.", chat_content, start_index)