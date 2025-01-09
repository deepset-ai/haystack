# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools.tool import Tool, ToolInvocationError, deserialize_tools_inplace, _check_duplicate_tool_names


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


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
            "type": "haystack.tools.tool.Tool",
            "data": {
                "name": "weather",
                "description": "Get weather report",
                "parameters": parameters,
                "function": "test_tool.get_weather_report",
            },
        }

    def test_from_dict(self):
        tool_dict = {
            "type": "haystack.tools.tool.Tool",
            "data": {
                "name": "weather",
                "description": "Get weather report",
                "parameters": parameters,
                "function": "test_tool.get_weather_report",
            },
        }

        tool = Tool.from_dict(tool_dict)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report


def test_deserialize_tools_inplace():
    tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

    data = {"tools": [tool.to_dict()]}
    deserialize_tools_inplace(data)
    assert data["tools"] == [tool]

    data = {"mytools": [tool.to_dict()]}
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

    # not a subclass of Tool
    data = {"tools": [{"type": "haystack.dataclasses.ChatMessage", "data": {"irrelevant": "irrelevant"}}]}
    with pytest.raises(TypeError):
        deserialize_tools_inplace(data)


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
