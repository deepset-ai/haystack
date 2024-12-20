# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

import pytest

from haystack.dataclasses.tool import (
    SchemaGenerationError,
    Tool,
    ToolInvocationError,
    _remove_title_from_schema,
    deserialize_tools_inplace,
    _check_duplicate_tool_names,
)

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


def function_with_docstring(city: str) -> str:
    """Get weather report for a city."""
    return f"Weather report for {city}: 20°C, sunny"


class TestTool:
    def test_init(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report

    def test_init_invalid_parameters(self):
        parameters = {"type": "invalid", "properties": {"city": {"type": "string"}}}

        with pytest.raises(ValueError):
            Tool(name="irrelevant", description="irrelevant", parameters=parameters, function=get_weather_report)

    def test_tool_spec(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        assert tool.tool_spec == {"name": "weather", "description": "Get weather report", "parameters": parameters}

    def test_invoke(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        assert tool.invoke(city="Berlin") == "Weather report for Berlin: 20°C, sunny"

    def test_invoke_fail(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        with pytest.raises(ToolInvocationError):
            tool.invoke()

    def test_to_dict(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        assert tool.to_dict() == {
            "name": "weather",
            "description": "Get weather report",
            "parameters": parameters,
            "function": "test_tool.get_weather_report",
        }

    def test_from_dict(self):
        tool_dict = {
            "name": "weather",
            "description": "Get weather report",
            "parameters": parameters,
            "function": "test_tool.get_weather_report",
        }

        tool = Tool.from_dict(tool_dict)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report

    def test_from_function_description_from_docstring(self):
        tool = Tool.from_function(function=function_with_docstring)

        assert tool.name == "function_with_docstring"
        assert tool.description == "Get weather report for a city."
        assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        assert tool.function == function_with_docstring

    def test_from_function_with_empty_description(self):
        tool = Tool.from_function(function=function_with_docstring, description="")

        assert tool.name == "function_with_docstring"
        assert tool.description == ""
        assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        assert tool.function == function_with_docstring

    def test_from_function_with_custom_description(self):
        tool = Tool.from_function(function=function_with_docstring, description="custom description")

        assert tool.name == "function_with_docstring"
        assert tool.description == "custom description"
        assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        assert tool.function == function_with_docstring

    def test_from_function_with_custom_name(self):
        tool = Tool.from_function(function=function_with_docstring, name="custom_name")

        assert tool.name == "custom_name"
        assert tool.description == "Get weather report for a city."
        assert tool.parameters == {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        assert tool.function == function_with_docstring

    def test_from_function_missing_type_hint(self):
        def function_missing_type_hint(city) -> str:
            return f"Weather report for {city}: 20°C, sunny"

        with pytest.raises(ValueError):
            Tool.from_function(function=function_missing_type_hint)

    def test_from_function_schema_generation_error(self):
        def function_with_invalid_type_hint(city: "invalid") -> str:
            return f"Weather report for {city}: 20°C, sunny"

        with pytest.raises(SchemaGenerationError):
            Tool.from_function(function=function_with_invalid_type_hint)

    def test_from_function_annotated(self):
        def function_with_annotations(
            city: Annotated[str, "the city for which to get the weather"] = "Munich",
            unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
            nullable_param: Annotated[Optional[str], "a nullable parameter"] = None,
        ) -> str:
            """A simple function to get the current weather for a location."""
            return f"Weather report for {city}: 20 {unit}, sunny"

        tool = Tool.from_function(function=function_with_annotations)

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


def test_deserialize_tools_inplace():
    tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)
    serialized_tool = tool.to_dict()
    print(serialized_tool)

    data = {"tools": [serialized_tool.copy()]}
    deserialize_tools_inplace(data)
    assert data["tools"] == [tool]

    data = {"mytools": [serialized_tool.copy()]}
    deserialize_tools_inplace(data, key="mytools")
    assert data["mytools"] == [tool]

    data = {"no_tools": 123}
    deserialize_tools_inplace(data)
    assert data == {"no_tools": 123}


def test_deserialize_tools_inplace_failures():
    data = {"key": "value"}
    deserialize_tools_inplace(data)
    assert data == {"key": "value"}

    data = {"tools": None}
    deserialize_tools_inplace(data)
    assert data == {"tools": None}

    data = {"tools": "not a list"}
    with pytest.raises(TypeError):
        deserialize_tools_inplace(data)

    data = {"tools": ["not a dictionary"]}
    with pytest.raises(TypeError):
        deserialize_tools_inplace(data)


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


def test_check_duplicate_tool_names():
    tools = [
        Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report),
        Tool(name="weather", description="A different description", parameters=parameters, function=get_weather_report),
    ]
    with pytest.raises(ValueError):
        _check_duplicate_tool_names(tools)

    tools = [
        Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report),
        Tool(name="weather2", description="Get weather report", parameters=parameters, function=get_weather_report),
    ]
    _check_duplicate_tool_names(tools)
