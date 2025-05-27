# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools.errors import SchemaGenerationError
from haystack.tools.from_function import create_tool_from_function, _remove_title_from_schema, tool
from typing import Annotated, Literal, Optional


def function_with_docstring(city: str) -> str:
    """Get weather report for a city."""
    return f"Weather report for {city}: 20°C, sunny"


def test_from_function_description_from_docstring():
    tool = create_tool_from_function(function=function_with_docstring)

    assert tool.name == "function_with_docstring"
    assert tool.description == "Get weather report for a city."
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_empty_description():
    tool = create_tool_from_function(function=function_with_docstring, description="")

    assert tool.name == "function_with_docstring"
    assert tool.description == ""
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_custom_description():
    tool = create_tool_from_function(function=function_with_docstring, description="custom description")

    assert tool.name == "function_with_docstring"
    assert tool.description == "custom description"
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_with_custom_name():
    tool = create_tool_from_function(function=function_with_docstring, name="custom_name")

    assert tool.name == "custom_name"
    assert tool.description == "Get weather report for a city."
    assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    assert tool.function == function_with_docstring


def test_from_function_annotated():
    def function_with_annotations(
        city: Annotated[str, "the city for which to get the weather"] = "Munich",
        unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
        nullable_param: Annotated[Optional[str], "a nullable parameter"] = None,
    ) -> str:
        """A simple function to get the current weather for a location."""
        return f"Weather report for {city}: 20 {unit}, sunny"

    tool = create_tool_from_function(function=function_with_annotations)

    assert tool.name == "function_with_annotations"
    assert tool.description == "A simple function to get the current weather for a location."
    assert tool.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "the city for which to get the weather", "default": "Munich"},
            "unit": {
                "type": "string",
                "enum": ["Celsius", "Fahrenheit"],
                "description": "the unit for the temperature",
                "default": "Celsius",
            },
            "nullable_param": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "a nullable parameter",
                "default": None,
            },
        },
    }


def test_from_function_missing_type_hint():
    def function_missing_type_hint(city) -> str:
        return f"Weather report for {city}: 20°C, sunny"

    with pytest.raises(ValueError):
        create_tool_from_function(function=function_missing_type_hint)


def test_from_function_schema_generation_error():
    def function_with_invalid_type_hint(city: "invalid") -> str:
        return f"Weather report for {city}: 20°C, sunny"

    with pytest.raises(SchemaGenerationError):
        create_tool_from_function(function=function_with_invalid_type_hint)


def test_tool_decorator():
    @tool
    def get_weather(city: str) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city}: 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get weather report for a city."
    assert get_weather.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert callable(get_weather.function)
    assert get_weather.function("Berlin") == "Weather report for Berlin: 20°C, sunny"


def test_tool_decorator_with_annotated_params():
    @tool
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({format} format): 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get weather report for a city."
    assert get_weather.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The target city", "default": "Berlin"},
            "format": {"type": "string", "enum": ["short", "long"], "description": "Output format", "default": "short"},
        },
    }
    assert callable(get_weather.function)
    assert get_weather.function("Berlin", "short") == "Weather report for Berlin (short format): 20°C, sunny"


def test_tool_decorator_with_parameters():
    @tool(name="fetch_weather", description="A tool to check the weather.")
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({format} format): 20°C, sunny"

    assert get_weather.name == "fetch_weather"
    assert get_weather.description == "A tool to check the weather."


def test_tool_decorator_with_inputs_and_outputs():
    @tool(inputs_from_state={"format": "format"}, outputs_to_state={"output": {"source": "output"}})
    def get_weather(
        city: Annotated[str, "The target city"] = "Berlin",
        format: Annotated[Literal["short", "long"], "Output format"] = "short",
    ) -> str:
        """Get weather report for a city."""
        return f"Weather report for {city} ({format} format): 20°C, sunny"

    assert get_weather.name == "get_weather"
    assert get_weather.inputs_from_state == {"format": "format"}
    assert get_weather.outputs_to_state == {"output": {"source": "output"}}
    # Inputs should be excluded from auto-generated parameters
    assert get_weather.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The target city", "default": "Berlin"}},
    }


def test_remove_title_from_schema():
    complex_schema = {
        "properties": {
            "parameter1": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "default": "default_value",
                "title": "Parameter1",
            },
            "parameter2": {
                "default": [1, 2, 3],
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "title": "Parameter2",
                "type": "array",
            },
            "parameter3": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "type": "array"},
                ],
                "default": 42,
                "title": "Parameter3",
            },
            "parameter4": {
                "anyOf": [{"type": "string"}, {"items": {"type": "integer"}, "type": "array"}, {"type": "object"}],
                "default": {"key": "value"},
                "title": "Parameter4",
            },
        },
        "title": "complex_function",
        "type": "object",
    }

    _remove_title_from_schema(complex_schema)

    assert complex_schema == {
        "properties": {
            "parameter1": {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "default_value"},
            "parameter2": {
                "default": [1, 2, 3],
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "type": "array",
            },
            "parameter3": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                    {"items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "type": "array"},
                ],
                "default": 42,
            },
            "parameter4": {
                "anyOf": [{"type": "string"}, {"items": {"type": "integer"}, "type": "array"}, {"type": "object"}],
                "default": {"key": "value"},
            },
        },
        "type": "object",
    }


def test_remove_title_from_schema_do_not_remove_title_property():
    """Test that the utility function only removes the 'title' keywords and not the 'title' property (if present)."""
    schema = {
        "properties": {
            "parameter1": {"type": "string", "title": "Parameter1"},
            "title": {"type": "string", "title": "Title"},
        },
        "title": "complex_function",
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {"properties": {"parameter1": {"type": "string"}, "title": {"type": "string"}}, "type": "object"}


def test_remove_title_from_schema_handle_no_title_in_top_level():
    schema = {
        "properties": {
            "parameter1": {"type": "string", "title": "Parameter1"},
            "parameter2": {"type": "integer", "title": "Parameter2"},
        },
        "type": "object",
    }

    _remove_title_from_schema(schema)

    assert schema == {
        "properties": {"parameter1": {"type": "string"}, "parameter2": {"type": "integer"}},
        "type": "object",
    }
