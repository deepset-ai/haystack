import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import yaml
from requests import RequestException

from haystack import Document, component, logging
from haystack.dataclasses.byte_stream import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install jsonref'") as openapi_imports:
    import jsonref


@component
class OpenAPIServiceToFunctions:
    """
    Converts OpenAPI service definitions to a format suitable for OpenAI function calling.

    The definition must respect OpenAPI specification 3.0.0 or higher.
    It can be specified in JSON or YAML format.
    Each function must have:
        - unique operationId
        - description
        - requestBody and/or parameters
        - schema for the requestBody and/or parameters
    For more details on OpenAPI specification see the [official documentation](https://github.com/OAI/OpenAPI-Specification).
    For more details on OpenAI function calling see the [official documentation](https://platform.openai.com/docs/guides/function-calling).

    Usage example:
    ```python
    from haystack.components.converters import OpenAPIServiceToFunctions

    converter = OpenAPIServiceToFunctions()
    result = converter.run(sources=["path/to/openapi_definition.yaml"])
    assert result["documents"]
    ```
    """

    MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3

    def __init__(self):
        """
        Create a OpenAPIServiceToFunctions component.
        """
        openapi_imports.check()

    @component.output_types(documents=List[Document])
    def run(
        self, sources: List[Union[str, Path, ByteStream]], system_messages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Converts OpenAPI definitions in OpenAI function calling format.

        :param sources:
            File paths, URLs or ByteStream objects of OpenAPI definitions.
        :param system_messages:
            Optional system messages for each source.

        :returns:
            A dictionary with the following keys:
            - documents: Documents containing a function definition and relevant metadata

        :raises RuntimeError:
            If the OpenAPI definitions cannot be downloaded or processed.
        :raises ValueError:
            If the source type is not recognized or no functions are found in the OpenAPI definitions.
        """
        documents: List[Document] = []
        system_messages = system_messages or [""] * len(sources)
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
                logger.warning(
                    "Invalid source type {source}. Only str, Path, and ByteStream are supported.", source=type(source)
                )
                continue

            if openapi_spec_content:
                try:
                    service_openapi_spec = self._parse_openapi_spec(openapi_spec_content)
                    functions: List[Dict[str, Any]] = self._openapi_to_functions(service_openapi_spec)
                    for function in functions:
                        meta: Dict[str, Any] = {"spec": service_openapi_spec}
                        if system_message:
                            meta["system_message"] = system_message
                        doc = Document(content=json.dumps(function), meta=meta)
                        documents.append(doc)
                except Exception as e:
                    logger.error(
                        "Error processing OpenAPI specification from source {source}: {error}", source=source, error=e
                    )

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
        for paths in service_openapi_spec["paths"].values():
            for path_spec in paths.values():
                function_dict = self._parse_endpoint_spec(path_spec)
                if function_dict:
                    functions.append(function_dict)
        return functions

    def _parse_endpoint_spec(self, resolved_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(resolved_spec, dict):
            logger.warning("Invalid OpenAPI spec format provided. Could not extract function.")
            return {}

        function_name = resolved_spec.get("operationId")
        description = resolved_spec.get("description") or resolved_spec.get("summary", "")

        schema: Dict[str, Any] = {"type": "object", "properties": {}}

        # requestBody section
        req_body_schema = (
            resolved_spec.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        )
        if "properties" in req_body_schema:
            for prop_name, prop_schema in req_body_schema["properties"].items():
                schema["properties"][prop_name] = self._parse_property_attributes(prop_schema)

            if "required" in req_body_schema:
                schema.setdefault("required", []).extend(req_body_schema["required"])

        # parameters section
        for param in resolved_spec.get("parameters", []):
            if "schema" in param:
                schema_dict = self._parse_property_attributes(param["schema"])
                # these attributes are not in param[schema] level but on param level
                useful_attributes = ["description", "pattern", "enum"]
                schema_dict.update({key: param[key] for key in useful_attributes if param.get(key)})
                schema["properties"][param["name"]] = schema_dict
                if param.get("required", False):
                    schema.setdefault("required", []).append(param["name"])

        if function_name and description and schema["properties"]:
            return {"name": function_name, "description": description, "parameters": schema}
        else:
            logger.warning(
                "Invalid OpenAPI spec format provided. Could not extract function from {spec}", spec=resolved_spec
            )
            return {}

    def _parse_property_attributes(
        self, property_schema: Dict[str, Any], include_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recursively parses the attributes of a property schema, including nested objects and arrays,
        and includes specified attributes like description, pattern, etc.

        :param property_schema: The schema of the property to parse.
        :param include_attributes: The list of attributes to include in the parsed schema.
        :return: The parsed schema of the property including the specified attributes.
        """
        include_attributes = include_attributes or ["description", "pattern", "enum"]

        schema_type = property_schema.get("type")

        parsed_schema = {"type": schema_type} if schema_type else {}
        for attr in include_attributes:
            if attr in property_schema:
                parsed_schema[attr] = property_schema[attr]

        if schema_type == "object":
            properties = property_schema.get("properties", {})
            parsed_properties = {
                prop_name: self._parse_property_attributes(prop, include_attributes)
                for prop_name, prop in properties.items()
            }
            parsed_schema["properties"] = parsed_properties

            if "required" in property_schema:
                parsed_schema["required"] = property_schema["required"]

        elif schema_type == "array":
            items = property_schema.get("items", {})
            parsed_schema["items"] = self._parse_property_attributes(items, include_attributes)

        return parsed_schema

    def _parse_openapi_spec(self, content: str) -> Dict[str, Any]:
        """
        Parses OpenAPI specification content, supporting both JSON and YAML formats.

        :param content: The content of the OpenAPI specification.
        :return: The parsed OpenAPI specification.
        """
        open_api_spec_content = None
        try:
            open_api_spec_content = json.loads(content)
            return jsonref.replace_refs(open_api_spec_content)
        except json.JSONDecodeError as json_error:
            # heuristic to confirm that the content is likely malformed JSON
            if content.strip().startswith(("{", "[")):
                raise json_error

        try:
            open_api_spec_content = yaml.safe_load(content)
        except yaml.YAMLError:
            error_message = (
                "Failed to parse the OpenAPI specification. "
                "The content does not appear to be valid JSON or YAML.\n\n"
            )
            raise RuntimeError(error_message, content)

        # Replace references in the object with their resolved values, if any
        return jsonref.replace_refs(open_api_spec_content)

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
            logger.warning("IO error reading file: {path}. Error: {error}", path=path, error=e)
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
            logger.warning("Error fetching URL: {url}. Error: {error}", url=url, error=e)
            return None
