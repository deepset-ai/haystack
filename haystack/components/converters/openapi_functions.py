import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import requests
import yaml
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
    OpenAPIServiceToFunctions is responsible for converting an OpenAPI service specification into a format suitable
    for OpenAI function calling, based on the provided OpenAPI specification. Given an OpenAPI specification,
    OpenAPIServiceToFunctions processes it, and extracts function definitions that can be invoked via OpenAI's
    function calling mechanism. The format of the extracted functions is compatible with OpenAI's function calling
    JSON format.

    Minimal requirements for OpenAPI specification:
    - OpenAPI version 3.0.0 or higher
    - Each function must have a unique operationId
    - Each function must have a description
    - Each function must have a requestBody or parameters or both
    - Each function must have a schema for the requestBody and/or parameters


    See https://github.com/OAI/OpenAPI-Specification for more details on OpenAPI specification.
    See https://platform.openai.com/docs/guides/function-calling for more details on OpenAI function calling.
    """

    MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3

    def __init__(self):
        """
        Initializes the OpenAPIServiceToFunctions instance
        """
        openapi_imports.check()

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], system_messages: List[str]) -> Dict[str, Any]:
        """
        Processes OpenAPI specification URLs or files to extract functions that can be invoked via OpenAI function
        calling mechanism. Each source is paired with a system message in one-to-one correspondence. The system message
        is used to assist LLM in the response generation.

        :param sources: A list of OpenAPI specification sources, which can be URLs, file paths, or ByteStream objects.
        :type sources: List[Union[str, Path, ByteStream]]
        :param system_messages: A list of system messages corresponding to each source.
        :type system_messages: List[str]
        :return: A dictionary with a key 'documents' containing a list of Document objects. Each Document object
                 encapsulates a function definition and relevant metadata.
        :rtype: Dict[str, Any]
        :raises RuntimeError: If the OpenAPI specification cannot be downloaded or processed.
        :raises ValueError: If the source type is not recognized or no functions are found in the OpenAPI specification.
        """
        documents: List[Document] = []
        for source, system_message in zip(sources, system_messages):
            openapi_spec_content = None
            if isinstance(source, (str, Path)):
                # check if the source is a file path or a URL
                if os.path.exists(source):
                    openapi_spec_content = self._read_from_file(source)
                else:
                    openapi_spec_content = self._read_from_url(str(source))
            elif isinstance(source, ByteStream):
                openapi_spec_content = source.data.decode("utf-8")
            else:
                logger.warning("Invalid source type %s. Only str, Path, and ByteStream are supported.", type(source))
                continue

            if openapi_spec_content:
                try:
                    service_openapi_spec = self._parse_openapi_spec(openapi_spec_content)
                    functions: List[Dict[str, Any]] = self._openapi_to_functions(service_openapi_spec)
                    docs = [
                        Document(
                            content=json.dumps(function),
                            meta={"spec": service_openapi_spec, "system_message": system_message},
                        )
                        for function in functions
                    ]
                    documents.extend(docs)
                except Exception as e:
                    logger.error("Error processing OpenAPI specification from source %s: %s", source, e)

        return {"documents": documents}

    def _openapi_to_functions(self, service_openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts functions from the OpenAPI specification of the service and converts them into a format
        suitable for OpenAI function calling.

        :param service_openapi_spec: The OpenAPI specification from which functions are to be extracted.
        :type service_openapi_spec: Dict[str, Any]
        :return: A list of dictionaries, each representing a function. Each dictionary includes the function's
                 name, description, and a schema of its parameters.
        :rtype: List[Dict[str, Any]]
        """

        # Doesn't enforce rigid spec validation because that would require a lot of dependencies
        # We check the version and require minimal fields to be present, so we can extract functions
        spec_version = service_openapi_spec.get("openapi")
        if not spec_version:
            raise ValueError(f"Invalid OpenAPI spec provided. Could not extract version from {service_openapi_spec}")
        service_openapi_spec_version = int(spec_version.split(".")[0])

        # Compare the versions
        if service_openapi_spec_version < OpenAPIServiceToFunctions.MIN_REQUIRED_OPENAPI_SPEC_VERSION:
            raise ValueError(
                f"Invalid OpenAPI spec version {service_openapi_spec_version}. Must be "
                f"at least {OpenAPIServiceToFunctions.MIN_REQUIRED_OPENAPI_SPEC_VERSION}."
            )

        functions: List[Dict[str, Any]] = []
        for path_methods in service_openapi_spec["paths"].values():
            for method_specification in path_methods.values():
                resolved_spec = jsonref.replace_refs(method_specification)
                if isinstance(resolved_spec, dict):
                    function_name = resolved_spec.get("operationId")
                    desc = resolved_spec.get("description") or resolved_spec.get("summary", "")

                    schema: Dict[str, Any] = {"type": "object", "properties": {}}

                    req_body = (
                        resolved_spec.get("requestBody", {})
                        .get("content", {})
                        .get("application/json", {})
                        .get("schema")
                    )
                    if req_body:
                        schema["properties"]["requestBody"] = req_body

                    params = resolved_spec.get("parameters", [])
                    if params:
                        param_properties = {param["name"]: param["schema"] for param in params if "schema" in param}
                        schema["properties"]["parameters"] = {"type": "object", "properties": param_properties}

                    # these three fields are minimal requirement for OpenAI function calling
                    if function_name and desc and schema:
                        functions.append({"name": function_name, "description": desc, "parameters": schema})
                    else:
                        logger.warning(
                            "Invalid OpenAPI spec format provided. Could not extract function from %s", resolved_spec
                        )

                else:
                    logger.warning(
                        "Invalid OpenAPI spec format provided. Could not extract function from %s", resolved_spec
                    )

        return functions

    def _parse_openapi_spec(self, content: str) -> Dict[str, Any]:
        """
        Parses OpenAPI specification content, supporting both JSON and YAML formats.

        :param content: The content of the OpenAPI specification.
        :type content: str
        :return: The parsed OpenAPI specification.
        :rtype: Dict[str, Any]
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as json_error:
            # heuristic to confirm that the content is likely malformed JSON
            if content.strip().startswith(("{", "[")):
                raise json_error

        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            error_message = (
                "Failed to parse the OpenAPI specification. "
                "The content does not appear to be valid JSON or YAML.\n\n"
            )
            raise RuntimeError(error_message, content)

    def _read_from_file(self, path: Union[str, Path]) -> Optional[str]:
        """
        Reads the content of a file, given its path.
        :param path: The path of the file.
        :type path: Union[str, Path]
        :return: The content of the file or None if the file cannot be read.
        """
        try:
            with open(path, "r") as f:
                return f.read()
        except IOError as e:
            logger.warning("IO error reading file: %s. Error: %s", path, e)
            return None

    def _read_from_url(self, url: str) -> Optional[str]:
        """
        Reads the content of a URL.
        :param url: The URL to read.
        :type url: str
        :return: The content of the URL or None if the URL cannot be read.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except RequestException as e:
            logger.warning("Error fetching URL: %s. Error: %s", url, e)
            return None
