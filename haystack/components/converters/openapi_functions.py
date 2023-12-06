import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union

from requests import RequestException

from haystack import component, Document
from haystack.dataclasses.byte_stream import ByteStream
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

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> Dict[str, Any]:
        """
        Processes an OpenAPI specification URL to extract functions that can be invoked via OpenAI function
        calling mechanism. It downloads the OpenAPI specification, processes it, and extracts function
        definitions. The extracted functions format is OpenAI function calling JSON.

        :param sources: A list of OpenAPI specification sources
        :type sources: List[Union[str, Path, ByteStream]]
        :return: A dictionary containing the extracted functions and the OpenAPI specification.
        :rtype: Dict[str, Any]
        :raises RuntimeError: If the OpenAPI specification cannot be downloaded or processed.
        :raises ValueError: If no functions are found in the OpenAPI specification.
        """
        documents: List[Document] = []
        for source in sources:
            try:
                if isinstance(source, (str, Path)):
                    if os.path.exists(source):
                        with open(source, "r") as f:
                            service_openapi_spec = jsonref.load(f)
                    else:
                        # Assume it is a URL
                        service_openapi_spec = jsonref.jsonloader(uri=source)
                elif isinstance(source, ByteStream):
                    service_openapi_spec = jsonref.loads(source.data.decode("utf-8"))
                else:
                    raise ValueError(f"Invalid source type {type(source)}")
                functions: List[Dict[str, Any]] = self._openapi_to_functions(service_openapi_spec)
                docs = [Document(content=str(function), meta={"spec": service_openapi_spec}) for function in functions]
                documents.extend(docs)
            except (RequestException, ValueError) as e:
                logger.warning(f"Could not download {source}. Skipping it. Error: {e}")
                continue

        return {"documents": documents}

    def _openapi_to_functions(self, service_openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
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
