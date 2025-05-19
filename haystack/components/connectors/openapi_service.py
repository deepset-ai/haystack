# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections import defaultdict
from copy import copy
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install openapi3'") as openapi_imports:
    import requests
    from openapi3 import OpenAPI
    from openapi3.errors import UnexpectedResponseError
    from openapi3.paths import Operation

    # Patch the request method to add support for the proper raw_response handling
    # If you see that https://github.com/Dorthu/openapi3/pull/124/
    # is merged, we can remove this patch - notify authors of this code
    def patch_request(
        self,
        base_url: str,
        *,
        data: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
        raw_response: bool = False,
        security: Optional[Dict[str, str]] = None,
        session: Optional[Any] = None,
        verify: Union[bool, str] = True,
    ) -> Optional[Any]:
        """
        Sends an HTTP request as described by this path.

        :param base_url: The URL to append this operation's path to when making
                         the call.
        :param data: The request body to send.
        :param parameters: The parameters used to create the path.
        :param raw_response: If true, return the raw response instead of validating
                             and exterpolating it.
        :param security: The security scheme to use, and the values it needs to
                         process successfully.
        :param session: A persistent request session.
        :param verify: If we should do an ssl verification on the request or not.
                       In case str was provided, will use that as the CA.
        :return: The response data, either raw or processed depending on raw_response flag.
        """
        # Set request method (e.g. 'GET')
        self._request = requests.Request(self.path[-1])

        # Set self._request.url to base_url w/ path
        self._request.url = base_url + self.path[-2]

        parameters = parameters or {}
        security = security or {}

        if security and self.security:
            security_requirement = None
            for scheme, value in security.items():
                security_requirement = None
                for r in self.security:
                    if r.name == scheme:
                        security_requirement = r
                        self._request_handle_secschemes(r, value)

            if security_requirement is None:
                err_msg = """No security requirement satisfied (accepts {}) \
                          """.format(", ".join(self.security.keys()))
                raise ValueError(err_msg)

        if self.requestBody:
            if self.requestBody.required and data is None:
                err_msg = "Request Body is required but none was provided."
                raise ValueError(err_msg)

            self._request_handle_body(data)

        self._request_handle_parameters(parameters)

        if session is None:
            session = self._session

        # send the prepared request
        result = session.send(self._request.prepare(), verify=verify)

        # spec enforces these are strings
        status_code = str(result.status_code)

        # find the response model in spec we received
        expected_response = None
        if status_code in self.responses:
            expected_response = self.responses[status_code]
        elif "default" in self.responses:
            expected_response = self.responses["default"]

        if expected_response is None:
            raise UnexpectedResponseError(result, self)

        # if we got back a valid response code (or there was a default) and no
        # response content was expected, return None
        if expected_response.content is None:
            return None

        content_type = result.headers["Content-Type"]
        if ";" in content_type:
            # if the content type that came in included an encoding, we'll ignore
            # it for now (requests has already parsed it for us) and only look at
            # the MIME type when determining if an expected content type was returned.
            content_type = content_type.split(";")[0].strip()

        expected_media = expected_response.content.get(content_type, None)

        # If raw_response is True, return the raw text or json based on content type
        if raw_response:
            if "application/json" in content_type:
                return result.json()
            return result.text

        if expected_media is None and "/" in content_type:
            # accept media type ranges in the spec. the most specific matching
            # type should always be chosen, but if we do not have a match here
            # a generic range should be accepted if one if provided
            # https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.1.md#response-object

            generic_type = content_type.split("/")[0] + "/*"
            expected_media = expected_response.content.get(generic_type, None)

        if expected_media is None:
            err_msg = """Unexpected Content-Type {} returned for operation {} \
                         (expected one of {})"""
            err_var = result.headers["Content-Type"], self.operationId, ",".join(expected_response.content.keys())

            raise RuntimeError(err_msg.format(*err_var))

        if content_type.lower() == "application/json":
            return expected_media.schema.model(result.json())

        raise NotImplementedError("Only application/json content type is supported")

    # Apply the patch
    Operation.request = patch_request


@component
class OpenAPIServiceConnector:
    """
    A component which connects the Haystack framework to OpenAPI services.

    The `OpenAPIServiceConnector` component connects the Haystack framework to OpenAPI services, enabling it to call
    operations as defined in the OpenAPI specification of the service.

    It integrates with `ChatMessage` dataclass, where the payload in messages is used to determine the method to be
    called and the parameters to be passed. The message payload should be an OpenAI JSON formatted function calling
    string consisting of the method name and the parameters to be passed to the method. The method name and parameters
    are then used to invoke the method on the OpenAPI service. The response from the service is returned as a
    `ChatMessage`.

    Before using this component, users usually resolve service endpoint parameters with a help of
    `OpenAPIServiceToFunctions` component.

    The example below demonstrates how to use the `OpenAPIServiceConnector` to invoke a method on a https://serper.dev/
    service specified via OpenAPI specification.

    Note, however, that `OpenAPIServiceConnector` is usually not meant to be used directly, but rather as part of a
    pipeline that includes the `OpenAPIServiceToFunctions` component and an `OpenAIChatGenerator` component using LLM
    with the function calling capabilities. In the example below we use the function calling payload directly, but in a
    real-world scenario, the function calling payload would usually be generated by the `OpenAIChatGenerator` component.

    Usage example:

    ```python
    import json
    import requests

    from haystack.components.connectors import OpenAPIServiceConnector
    from haystack.dataclasses import ChatMessage


    fc_payload = [{'function': {'arguments': '{"q": "Why was Sam Altman ousted from OpenAI?"}', 'name': 'search'},
                   'id': 'call_PmEBYvZ7mGrQP5PUASA5m9wO', 'type': 'function'}]

    serper_token = <your_serper_dev_token>
    serperdev_openapi_spec = json.loads(requests.get("https://bit.ly/serper_dev_spec").text)
    service_connector = OpenAPIServiceConnector()
    result = service_connector.run(messages=[ChatMessage.from_assistant(json.dumps(fc_payload))],
                                   service_openapi_spec=serperdev_openapi_spec, service_credentials=serper_token)
    print(result)

    >> {'service_response': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
    >> '{"searchParameters": {"q": "Why was Sam Altman ousted from OpenAI?",
    >> "type": "search", "engine": "google"}, "answerBox": {"snippet": "Concerns over AI safety and OpenAI\'s role
    >> in protecting were at the center of Altman\'s brief ouster from the company."...
    ```

    """

    def __init__(self, ssl_verify: Optional[Union[bool, str]] = None):
        """
        Initializes the OpenAPIServiceConnector instance

        :param ssl_verify: Decide if to use SSL verification to the requests or not,
        in case a string is passed, will be used as the CA.
        """
        openapi_imports.check()
        self.ssl_verify = ssl_verify

    @component.output_types(service_response=Dict[str, Any])
    def run(
        self,
        messages: List[ChatMessage],
        service_openapi_spec: Dict[str, Any],
        service_credentials: Optional[Union[dict, str]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Processes a list of chat messages to invoke a method on an OpenAPI service.

        It parses the last message in the list, expecting it to contain tool calls.

        :param messages: A list of `ChatMessage` objects containing the messages to be processed. The last message
        should contain the tool calls.
        :param service_openapi_spec: The OpenAPI JSON specification object of the service to be invoked. All the refs
        should already be resolved.
        :param service_credentials: The credentials to be used for authentication with the service.
        Currently, only the http and apiKey OpenAPI security schemes are supported.

        :return: A dictionary with the following keys:
            - `service_response`:  a list of `ChatMessage` objects, each containing the response from the service. The
                                   response is in JSON format, and the `content` attribute of the `ChatMessage` contains
                                   the JSON string.

        :raises ValueError: If the last message is not from the assistant or if it does not contain tool calls.
        """

        last_message = messages[-1]
        if not last_message.is_from(ChatRole.ASSISTANT):
            raise ValueError(f"{last_message} is not from the assistant.")

        tool_calls = last_message.tool_calls
        if not tool_calls:
            raise ValueError(f"The provided ChatMessage has no tool calls.\nChatMessage: {last_message}")

        function_payloads = []
        for tool_call in tool_calls:
            function_payloads.append({"arguments": tool_call.arguments, "name": tool_call.tool_name})

        # instantiate the OpenAPI service for the given specification
        openapi_service = OpenAPI(service_openapi_spec, ssl_verify=self.ssl_verify)
        self._authenticate_service(openapi_service, service_credentials)

        response_messages = []
        for method_invocation_descriptor in function_payloads:
            service_response = self._invoke_method(openapi_service, method_invocation_descriptor)
            # openapi3 parses the JSON service response into a model object, which is not our focus at the moment.
            # Instead, we require direct access to the raw JSON data of the response, rather than the model objects
            # provided by the openapi3 library. This approach helps us avoid issues related to (de)serialization.
            # By accessing the raw JSON response through `service_response._raw_data`, we can serialize this data
            # into a string. Finally, we use this string to create a ChatMessage object.
            response_messages.append(ChatMessage.from_user(json.dumps(service_response)))

        return {"service_response": response_messages}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, ssl_verify=self.ssl_verify)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAPIServiceConnector":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def _authenticate_service(self, openapi_service: "OpenAPI", credentials: Optional[Union[dict, str]] = None):
        """
        Authentication with an OpenAPI service.

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
        :param credentials: Credentials for authentication, which can be either a string (e.g. token) or a dictionary
        with keys matching the authentication method names.
        :raises ValueError: If authentication fails, is not found, or if appropriate credentials are missing.
        """
        if openapi_service.raw_element.get("components", {}).get("securitySchemes"):
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
                        break

                    raise ValueError(
                        f"Service {service_name} requires {scheme_name} security scheme but no "
                        f"credentials were provided for it. Check the service configuration and credentials."
                    )
            if not authenticated:
                raise ValueError(
                    f"Service {service_name} requires authentication but no credentials were provided "
                    f"for it. Check the service configuration and credentials."
                )

    def _invoke_method(self, openapi_service: "OpenAPI", method_invocation_descriptor: Dict[str, Any]) -> Any:
        """
        Invokes the specified method on the OpenAPI service.

        The method name and arguments are passed in the method_invocation_descriptor.

        :param openapi_service: The OpenAPI service instance.
        :param method_invocation_descriptor: The method name and arguments to be passed to the method. The payload
        should contain the method name (key: "name") and the arguments (key: "arguments"). The name is a string, and
        the arguments are a dictionary of key-value pairs.
        :return: A service JSON response.
        :raises RuntimeError: If the method is not found or invocation fails.
        """
        name = method_invocation_descriptor.get("name")
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
        operation_dict = operation.raw_element

        # Pack URL/query parameters under "parameters" key
        method_call_params: Dict[str, Dict[str, Any]] = defaultdict(dict)
        parameters = operation_dict.get("parameters", [])
        request_body = operation_dict.get("requestBody", {})

        for param in parameters:
            param_name = param["name"]
            param_value = invocation_arguments.get(param_name)
            if param_value:
                method_call_params["parameters"][param_name] = param_value
            else:
                if param.get("required", False):
                    raise ValueError(f"Missing parameter: '{param_name}' required for the '{name}' operation.")

        # Pack request body parameters under "data" key
        if request_body:
            schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})
            required_params = schema.get("required", [])
            for param_name in schema.get("properties", {}):
                param_value = invocation_arguments.get(param_name)
                if param_value:
                    method_call_params["data"][param_name] = param_value
                else:
                    if param_name in required_params:
                        raise ValueError(
                            f"Missing requestBody parameter: '{param_name}' required for the '{name}' operation."
                        )
        # call the underlying service REST API with the parameters
        return method_to_call(**method_call_params, raw_response=True)
