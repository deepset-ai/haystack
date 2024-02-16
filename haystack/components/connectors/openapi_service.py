import json
import logging
from collections import defaultdict
from copy import copy
from typing import List, Dict, Any, Optional, Union

from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install openapi3'") as openapi_imports:
    from openapi3 import OpenAPI
    from openapi3.paths import Operation


@component
class OpenAPIServiceConnector:
    """
    OpenAPIServiceConnector connects to OpenAPI services, allowing for the invocation of methods specified in
    an OpenAPI specification of that service. It integrates with ChatMessage interface, where messages are used to
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
    def run(
        self,
        messages: List[ChatMessage],
        service_openapi_spec: Dict[str, Any],
        service_credentials: Optional[Union[dict, str]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Processes a list of chat messages to invoke a method on an OpenAPI service. It parses the last message in the
        list, expecting it to contain an OpenAI function calling descriptor (name & parameters) in JSON format.

        :param messages: A list of `ChatMessage` objects representing the chat history.
        :type messages: List[ChatMessage]
        :param service_openapi_spec: The OpenAPI JSON specification object of the service.
        :type service_openapi_spec: JSON object
        :return: A dictionary with a key `"service_response"`, containing the response from the OpenAPI service.
        :rtype: Dict[str, List[ChatMessage]]
        :param service_credentials: The credentials to be used for authentication with the service.
        Currently, only the http and apiKey schemes are supported. See _authenticate_service method for more details.
        :type service_credentials: Optional[Union[dict, str]]
        :raises ValueError: If the last message is not from the assistant or if it does not contain the correct payload
        to invoke a method on the service.
        """

        last_message = messages[-1]
        if not last_message.is_from(ChatRole.ASSISTANT):
            raise ValueError(f"{last_message} is not from the assistant.")

        function_invocation_payloads = self._parse_message(last_message)

        # instantiate the OpenAPI service for the given specification
        openapi_service = OpenAPI(service_openapi_spec)
        self._authenticate_service(openapi_service, service_credentials)

        response_messages = []
        for method_invocation_descriptor in function_invocation_payloads:
            service_response = self._invoke_method(openapi_service, method_invocation_descriptor)
            # openapi3 parses the JSON service response into a model object, which is not our focus at the moment.
            # Instead, we require direct access to the raw JSON data of the response, rather than the model objects
            # provided by the openapi3 library. This approach helps us avoid issues related to (de)serialization.
            # By accessing the raw JSON response through `service_response._raw_data`, we can serialize this data
            # into a string. Finally, we use this string to create a ChatMessage object.
            response_messages.append(ChatMessage.from_user(json.dumps(service_response._raw_data)))

        return {"service_response": response_messages}

    def _parse_message(self, message: ChatMessage) -> List[Dict[str, Any]]:
        """
        Parses the message to extract the method invocation descriptor.

        :param message: ChatMessage containing the tools calls
        :type message: ChatMessage
        :return: A list of function invocation payloads
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If the content is not valid JSON or lacks required fields.
        """
        function_payloads = []
        try:
            tool_calls = json.loads(message.content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON content, expected OpenAI tools message.", message.content)

        for tool_call in tool_calls:
            # this should never happen, but just in case do a sanity check
            if "type" not in tool_call:
                raise ValueError("Message payload doesn't seem to be a tool invocation descriptor", message.content)

            # In OpenAPIServiceConnector we know how to handle functions tools only
            if tool_call["type"] == "function":
                function_call = tool_call["function"]
                function_payloads.append(
                    {"arguments": json.loads(function_call["arguments"]), "name": function_call["name"]}
                )
        return function_payloads

    def _authenticate_service(self, openapi_service: OpenAPI, credentials: Optional[Union[dict, str]] = None):
        """
        Authenticates with the OpenAPI service if required, supporting both single (str) and multiple
        authentication methods (dict).

        OpenAPI spec v3 supports the following security schemes:
        http – for Basic, Bearer and other HTTP authentications schemes
        apiKey – for API keys and cookie authentication
        oauth2 – for OAuth 2
        openIdConnect – for OpenID Connect Discovery

        Currently, only the http and apiKey schemes are supported. Multiple security schemes can be defined in the
        OpenAPI spec, and the credentials should be provided as a dictionary with keys matching the security scheme
        names. If only one security scheme is defined, the credentials can be provided as a simple string.

        :param openapi_service: The OpenAPI service instance.
        :type openapi_service: OpenAPI
        :param credentials: Credentials for authentication, which can be either a string (e.g. token) or a dictionary
        with keys matching the authentication method names.
        :type credentials: dict | str, optional
        :raises ValueError: If authentication fails, is not found, or if appropriate credentials are missing.
        """
        if self._has_security_schemes(openapi_service):
            service_name = openapi_service.info.title
            if not credentials:
                raise ValueError(f"Service {service_name} requires authentication but no credentials were provided.")

            # a dictionary of security schemes defined in the OpenAPI spec
            # each key is the name of the security scheme, and the value is the scheme definition
            security_schemes = openapi_service.components.securitySchemes.raw_element
            supported_schemes = ["http", "apiKey"]  # todo: add support for oauth2 and openIdConnect

            authenticated = False
            for scheme_name, scheme in security_schemes.items():
                if scheme["type"] in supported_schemes:
                    auth_credentials = None
                    if isinstance(credentials, str):
                        auth_credentials = credentials
                    elif isinstance(credentials, dict) and scheme_name in credentials:
                        auth_credentials = credentials[scheme_name]
                    if auth_credentials:
                        openapi_service.authenticate(scheme_name, auth_credentials)
                        authenticated = True
                    else:
                        raise ValueError(
                            f"Service {service_name} requires {scheme_name} security scheme but no "
                            f"credentials were provided for it. Check the service configuration and credentials."
                        )
            if not authenticated:
                raise ValueError(
                    f"Service {service_name} requires authentication but no credentials were provided "
                    f"for it. Check the service configuration and credentials."
                )

    def _invoke_method(self, openapi_service: OpenAPI, method_invocation_descriptor: Dict[str, Any]) -> Any:
        """
        Invokes the specified method on the OpenAPI service. The method name and arguments are passed in the
        method_invocation_descriptor.

        :param openapi_service: The OpenAPI service instance.
        :type openapi_service: OpenAPI
        :param method_invocation_descriptor: The method name and arguments to be passed to the method. The payload
        should contain the method name (key: "name") and the arguments (key: "arguments"). The name is a string, and
        the arguments are a dictionary of key-value pairs.
        :type method_invocation_descriptor: Dict[str, Any]
        :return: A service JSON response.
        :rtype: Any
        :raises RuntimeError: If the method is not found or invocation fails.
        """
        name = method_invocation_descriptor.get("name", None)
        invocation_arguments = copy(method_invocation_descriptor.get("arguments", {}))
        if not name or not invocation_arguments:
            raise ValueError(
                f"Invalid function calling descriptor: {method_invocation_descriptor} . It should contain "
                f"a method name and arguments."
            )

        # openapi3 specific method to call the operation, do we have it?
        method_to_call = getattr(openapi_service, f"call_{name}", None)
        if not callable(method_to_call):
            raise RuntimeError(f"Operation {name} not found in OpenAPI specification {openapi_service.info.title}")

        # get the operation reference from the method_to_call
        operation = method_to_call.operation.__self__

        # Pack URL/query parameters under "parameters" key
        method_call_params: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for param_name in self._parameter_names(operation):
            method_call_params["parameters"][param_name] = invocation_arguments.pop(param_name, "")

        # Pack request body parameters under "data" key
        if self._has_request_body(operation):
            for param_name in self._body_request_parameter_names(operation):
                method_call_params["data"][param_name] = invocation_arguments.pop(param_name, "")

        # call the underlying service REST API with the parameters
        return method_to_call(**method_call_params)

    def _has_security_schemes(self, openapi_service: OpenAPI) -> bool:
        """
        Checks if the OpenAPI service has security schemes defined.

        :param openapi_service: The OpenAPI service instance.
        :type openapi_service: OpenAPI
        :return: True if the service has security schemes defined, False otherwise.
        :rtype: bool
        """
        return bool(self._safe_get_nested_keys(openapi_service.raw_element, ["components", "securitySchemes"]))

    def _has_parameters(self, openapi_op: Operation) -> bool:
        """
        Checks if the OpenAPI operation has parameters defined.

        :param openapi_op: The OpenAPI operation instance.
        :type openapi_op: Operation
        :return: True if the operation has parameters defined, False otherwise.
        :rtype: bool
        """
        return bool(openapi_op.raw_element.get("parameters"))

    def _has_request_body(self, openapi_op: Operation) -> bool:
        """
        Checks if the OpenAPI operation has a request body defined.

        :param openapi_op: The OpenAPI operation instance.
        :type openapi_op: Operation
        :return: True if the operation has a request body defined, False otherwise.
        :rtype: bool
        """
        return bool(openapi_op.raw_element.get("requestBody"))

    def _parameter_names(self, openapi_op: Operation) -> List[str]:
        """
        Extracts the parameter names from the OpenAPI operation.
        :param openapi_op: The OpenAPI operation instance.
        :type openapi_op: Operation
        :return: A list of parameter names.

        """
        return [param.get("name", "") for param in openapi_op.raw_element.get("parameters", [])]

    def _body_request_parameter_names(self, openapi_op: Operation) -> List[str]:
        """
        Extracts the parameter names from the request body of the OpenAPI operation.
        """
        keys_path = ["requestBody", "content", "application/json", "schema", "properties"]
        return self._safe_get_nested_keys(openapi_op.raw_element, keys_path)

    def _safe_get_nested_keys(self, d, keys):
        """
        Safely get nested keys from a dictionary

        :param d: The dictionary from which to fetch the value.
        :param keys: A list of keys representing the path to the desired value.
        :return: The keys of the nested dictionary if present, otherwise an empty list.
        """
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return []
        return list(d.keys()) if isinstance(d, dict) else []
