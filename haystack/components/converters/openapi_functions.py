import logging
from typing import List, Dict, Any

import requests
from requests import RequestException

from haystack import component
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install jsonref'") as openapi_imports:
    import jsonref


@component
class OpenAPIServiceToFunctions:
    """
    OpenAPIServiceToFunctions converts OpenAPI service specification to OpenAI function calling JSON format
    from a given OpenAPI specification URL. It fetches the OpenAPI specification, processes it, and extracts
    function definitions that can be invoked via OpenAI function calling mechanism. The extracted functions
    format is OpenAI function calling JSON.

    See https://github.com/OAI/OpenAPI-Specification for more details on OpenAPI specification.
    See https://platform.openai.com/docs/guides/function-calling for more details on OpenAI function calling.
    """

    def __init__(self):
        """
        Initializes the OpenAPIServiceToFunctions instance
        """
        openapi_imports.check()

    def openapi_to_functions(self, service_openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract functions from the OpenAPI specification of the service and convert them to OpenAI function calling
        JSON format.

        :param service_openapi_spec: The OpenAPI specification from which functions are to be extracted.
        :type service_openapi_spec: Dict[str, Any]
        :return: A list of dictionaries, each representing a function with its name, description, and
        parameters schema.
        :rtype: List[Dict[str, Any]]
        """
        functions: List[Dict[str, Any]] = []
        for path_methods in service_openapi_spec["paths"].values():
            for method_specification in path_methods.values():
                resolved_spec = jsonref.replace_refs(method_specification)
                function_name = resolved_spec.get("operationId")
                desc = resolved_spec.get("description") or resolved_spec.get("summary", "")

                schema = {"type": "object", "properties": {}}

                req_body = (
                    resolved_spec.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
                )
                if req_body:
                    schema["properties"]["requestBody"] = req_body

                params = resolved_spec.get("parameters", [])
                if params:
                    param_properties = {param["name"]: param["schema"] for param in params if "schema" in param}
                    schema["properties"]["parameters"] = {"type": "object", "properties": param_properties}

                functions.append({"name": function_name, "description": desc, "parameters": schema})
        return functions

    @component.output_types(functions=Dict[str, Any], service_openapi_spec=Dict[str, Any])
    def run(self, service_spec_url: str) -> Dict[str, Any]:
        """
        Processes an OpenAPI specification URL to extract functions that can be invoked via OpenAI function
        calling mechanism. It downloads the OpenAPI specification, processes it, and extracts function
        definitions. The extracted functions format is OpenAI function calling JSON.

        :param service_spec_url: URL of the OpenAPI specification.
        :type service_spec_url: str
        :return: A dictionary containing the extracted functions and the OpenAPI specification.
        :rtype: Dict[str, Any]
        :raises RuntimeError: If the OpenAPI specification cannot be downloaded or processed.
        :raises ValueError: If no functions are found in the OpenAPI specification.
        """
        try:
            response = requests.get(service_spec_url)
            response.raise_for_status()
            logger.info(f"Successfully retrieved OpenAPI specification from {service_spec_url}")

        except RequestException as e:
            logger.error(f"Failed to download OpenAPI specification from {service_spec_url}: {e}")
            raise RuntimeError(f"Error downloading OpenAPI specification: {e}") from e

        try:
            service_openapi_spec = jsonref.loads(response.content)
        except Exception as e:
            logger.error(f"Failed to parse OpenAPI specification from {service_spec_url}: {e}")
            raise RuntimeError(f"Error parsing OpenAPI specification: {e}") from e

        openapi_functions = self.openapi_to_functions(service_openapi_spec)
        if not openapi_functions:
            logger.warning(f"No functions found in the OpenAPI specification from {service_spec_url}")
            raise ValueError(f"No functions found in the OpenAPI specification from {service_spec_url}")

        return {"functions": {"functions": openapi_functions}, "service_openapi_spec": service_openapi_spec}
