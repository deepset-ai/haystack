import pytest
from typing import Any, Dict, List, Union, Optional

from haystack.tools.property_schema_utils import _create_property_schema


@pytest.mark.parametrize(
    "python_type, expected_schema",
    [
        (str, {"type": "string", "description": ""}),
        (int, {"type": "integer", "description": ""}),
        (float, {"type": "number", "description": ""}),
        (bool, {"type": "boolean", "description": ""}),
        # TODO Check if LLM providers can handle this schema
        (list, {"type": "array", "description": ""}),
        # TODO Check if LLM providers can handle this schema (e.g. additionalProperties and or empty object)
        (dict, {"type": "object", "description": "", "additionalProperties": True}),
    ],
)
def test_create_property_schema_bare_types(python_type, expected_schema):
    """
    Test the _create_property_schema function with various Python types.
    """
    schema = _create_property_schema(python_type, "")
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (Optional[str], "An optional string", {"type": "string", "description": "An optional string"}),
        (Optional[int], "An optional integer", {"type": "integer", "description": "An optional integer"}),
        (Optional[float], "An optional float", {"type": "number", "description": "An optional float"}),
        (Optional[bool], "An optional boolean", {"type": "boolean", "description": "An optional boolean"}),
        (Optional[list], "An optional list", {"type": "array", "description": "An optional list"}),
        (
            Optional[dict],
            "An optional dict",
            {"type": "object", "description": "An optional dict", "additionalProperties": True},
        ),
    ],
)
def test_create_property_schema_optional_types(python_type, description, expected_schema):
    """
    Should produce the same schema as the bare type because the optional type is handled through the required field.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            List[str],
            "A list of strings",
            {"type": "array", "description": "A list of strings", "items": {"type": "string"}},
        ),
        (
            List[int],
            "A list of integers",
            {"type": "array", "description": "A list of integers", "items": {"type": "integer"}},
        ),
        (
            List[float],
            "A list of floats",
            {"type": "array", "description": "A list of floats", "items": {"type": "number"}},
        ),
        (
            List[bool],
            "A list of booleans",
            {"type": "array", "description": "A list of booleans", "items": {"type": "boolean"}},
        ),
    ],
)
def test_create_property_schema_list_of_types(python_type, description, expected_schema):
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            Union[str, int],
            "A union of string and integer",
            {"description": "A union of string and integer", "oneOf": [{"type": "string"}, {"type": "integer"}]},
        ),
        (
            Union[str, float],
            "A union of string and float",
            {"description": "A union of string and float", "oneOf": [{"type": "string"}, {"type": "number"}]},
        ),
    ],
)
def test_create_property_schema_union_type(python_type, description, expected_schema):
    """
    Test the _create_property_schema function with union types.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "python_type, description, expected_schema",
    [
        (
            Union[Dict[str, Any], List[Dict[str, Any]]],
            "Often found as the runtime param `meta` in our components",
            {
                "description": "Often found as the runtime param `meta` in our components",
                "oneOf": [
                    {"type": "object", "additionalProperties": True},
                    {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                ],
            },
        )
    ],
)
def test_create_property_schema_complex_types(python_type, description, expected_schema):
    """
    Tests for complex types, especially those found in our pre-built components.
    """
    schema = _create_property_schema(python_type, description)
    assert schema == expected_schema
