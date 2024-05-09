# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from haystack import component, logging
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
    assert result["functions"]
    ```
    """

    MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3

    def __init__(self):
        """
        Create an OpenAPIServiceToFunctions component.
        """
        openapi_imports.check()

    @component.output_types(functions=List[Dict[str, Any]], openapi_specs=List[Dict[str, Any]])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> Dict[str, Any]:
        """
        Converts OpenAPI definitions in OpenAI function calling format.

        :param sources:
            File paths or ByteStream objects of OpenAPI definitions (in JSON or YAML format).

        :returns:
            A dictionary with the following keys:
            - functions: Function definitions in JSON object format
            - openapi_specs: OpenAPI specs in JSON/YAML object format with resolved references

        :raises RuntimeError:
            If the OpenAPI definitions cannot be downloaded or processed.
        :raises ValueError:
            If the source type is not recognized or no functions are found in the OpenAPI definitions.
        """
        all_extracted_fc_definitions: List[Dict[str, Any]] = []
        all_openapi_specs = []
        for source in sources:
            openapi_spec_content = None
            if isinstance(source, (str, Path)):
                if os.path.exists(source):
                    try:
                        with open(source, "r") as f:
                            openapi_spec_content = f.read()
                    except IOError as e:
                        logger.warning(
                            "IO error reading OpenAPI specification file: {source}. Error: {e}", source=source, e=e
                        )
                else:
                    logger.warning(f"OpenAPI specification file not found: {source}")
            elif isinstance(source, ByteStream):
                openapi_spec_content = source.data.decode("utf-8")
                if not openapi_spec_content:
                    logger.warning(
                        "Invalid OpenAPI specification content provided: {openapi_spec_content}",
                        openapi_spec_content=openapi_spec_content,
                    )
            else:
                logger.warning(
                    "Invalid source type {source}. Only str, Path, and ByteStream are supported.", source=type(source)
                )
                continue

            if openapi_spec_content:
                try:
                    service_openapi_spec = self._parse_openapi_spec(openapi_spec_content)
                    functions: List[Dict[str, Any]] = self._openapi_to_functions(service_openapi_spec)
                    all_extracted_fc_definitions.extend(functions)
                    all_openapi_specs.append(service_openapi_spec)
                except Exception as e:
                    logger.error(
                        "Error processing OpenAPI specification from source {source}: {error}", source=source, error=e
                    )

        if not all_extracted_fc_definitions:
            logger.warning("No OpenAI function definitions extracted from the provided OpenAPI specification sources.")

        return {"functions": all_extracted_fc_definitions, "openapi_specs": all_openapi_specs}

    def _openapi_to_functions(self, service_openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        OpenAPI to OpenAI function conversion.

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
        Parses the attributes of a property schema.

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
