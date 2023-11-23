import json
import logging
from typing import List, Dict, Any


from haystack.dataclasses import ChatMessage, ChatRole
from haystack import component
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install openapi3'") as openapi_imports:
    from openapi3 import OpenAPI


@component
class OpenAPIServiceConnector:
    """
    OpenAPIServiceConnector connects to OpenAPI services, allowing for the invocation of methods specified in
    an OpenAPI specification of that service. It integrates it ChatMessage interface, where messages are used to
    determine the method to be called and the parameters to be passed. The message payload should be a JSON formatted
    string consisting of the method name and the parameters to be passed to the method. The method name and parameters
    are then used to invoke the method on the OpenAPI service. The response from the service is returned as a
    ChatMessage.

    Before using this component, one needs to register functions from the OpenAPI specification with LLM.
    This can be done using the OpenAPIServiceToFunctions component.
    """

    def __init__(self):
        """
        Initializes the OpenAPIServiceConnector instance
        """
        openapi_imports.check()

    @component.output_types(service_response=Dict[str, Any])
    def run(self, messages: List[ChatMessage], service_openapi_spec: Any):
        """
        Processes a list of chat messages to invoke a method on an OpenAPI service. It parses the last message in the
        list, expecting it to contain an OpenAI function calling descriptor (name & parameters) in JSON format.

        :param messages: A list of ``ChatMessage`` objects representing the chat history.
        :type messages: List[ChatMessage]
        :param service_openapi_spec: The OpenAPI JSON specification object of the service.
        :type service_openapi_spec: JSON object
        :return: A dictionary with a key ``"service_response"``, containing the response from the OpenAPI service.
        :rtype: Dict[str, Any]
        :raises ValueError: If the last message is not from the assistant or if it does not contain the correct payload
        to invoke a method on the service.
        """

        last_message = messages[-1]
        if last_message.is_from(ChatRole.ASSISTANT) and self._is_valid_json(last_message.content):
            method_invocation_descriptor = json.loads(last_message.content)
            parameters = json.loads(method_invocation_descriptor["arguments"])
            name = method_invocation_descriptor["name"]

            openapi_service = OpenAPI(service_openapi_spec)

            method_name = f"call_{name}"
            method_to_call = getattr(openapi_service, method_name, None)

            # Check if the method exists and then call it
            if callable(method_to_call):
                service_response = method_to_call(parameters=parameters["parameters"])
                return {"service_response": [ChatMessage.from_user(str(service_response))]}
            else:
                raise RuntimeError(f"Method {method_name} not found in the OpenAPI specification.")
        else:
            raise ValueError(
                f"Message {last_message} does not contain function calling payload to invoke a method on the service."
            )

    def _is_valid_json(self, content: str):
        """
        Validates whether the provided content is a valid JSON string.

        :param content: The string content to be checked.
        :type content: str
        :return: ``True`` if the content is a valid JSON string, ``False`` otherwise.
        :rtype: bool
        """
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False
