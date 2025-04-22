# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport("Run 'pip install openapi-llm'") as openapi_llm_imports:
    from openapi_llm.client.openapi import OpenAPIClient


@component
class OpenAPIConnector:
    """
    OpenAPIConnector enables direct invocation of REST endpoints defined in an OpenAPI specification.

    The OpenAPIConnector serves as a bridge between Haystack pipelines and any REST API that follows
    the OpenAPI(formerly Swagger) specification. It dynamically interprets the API specification and
    provides an interface for executing API operations. It is usually invoked by passing input
    arguments to it from a Haystack pipeline run method or by other components in a pipeline that
    pass input arguments to this component.

    Example:
    ```python
    from haystack.utils import Secret
    from haystack.components.connectors.openapi import OpenAPIConnector

    connector = OpenAPIConnector(
        openapi_spec="https://bit.ly/serperdev_openapi",
        credentials=Secret.from_env_var("SERPERDEV_API_KEY"),
        service_kwargs={"config_factory": my_custom_config_factory}
    )
    response = connector.run(
        operation_id="search",
        arguments={"q": "Who was Nikola Tesla?"}
    )
    ```
    Note:
    - The `parameters` argument is required for this component.
    - The `service_kwargs` argument is optional, it can be used to pass additional options to the OpenAPIClient.

    """

    def __init__(
        self, openapi_spec: str, credentials: Optional[Secret] = None, service_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the OpenAPIConnector with a specification and optional credentials.

        :param openapi_spec: URL, file path, or raw string of the OpenAPI specification
        :param credentials: Optional API key or credentials for the service wrapped in a Secret
        :param service_kwargs: Additional keyword arguments passed to OpenAPIClient.from_spec()
            For example, you can pass a custom config_factory or other configuration options.
        """
        openapi_llm_imports.check()
        self.openapi_spec = openapi_spec
        self.credentials = credentials
        self.service_kwargs = service_kwargs or {}

        self.client = OpenAPIClient.from_spec(
            openapi_spec=openapi_spec,
            credentials=credentials.resolve_value() if credentials else None,
            **self.service_kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            openapi_spec=self.openapi_spec,
            credentials=self.credentials.to_dict() if self.credentials else None,
            service_kwargs=self.service_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAPIConnector":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["credentials"])
        return default_from_dict(cls, data)

    @component.output_types(response=Dict[str, Any])
    def run(self, operation_id: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invokes a REST endpoint specified in the OpenAPI specification.

        :param operation_id: The operationId from the OpenAPI spec to invoke
        :param arguments: Optional parameters for the endpoint (query, path, or body parameters)
        :return: Dictionary containing the service response
        """
        payload = {"name": operation_id, "arguments": arguments or {}}

        # Invoke the endpoint using openapi-llm client
        response = self.client.invoke(payload)
        return {"response": response}
