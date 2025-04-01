# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
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
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_init_invalid_parameters(self):
        params = {"type": "invalid", "properties": {"city": {"type": "string"}}}
        with pytest.raises(ValueError):
            Tool(name="irrelevant", description="irrelevant", parameters=params, function=get_weather_report)

    @pytest.mark.parametrize(
        "outputs_to_state",
        [
            pytest.param({"documents": ["some_value"]}, id="config-not-a-dict"),
            pytest.param({"documents": {"source": get_weather_report}}, id="source-not-a-string"),
            pytest.param({"documents": {"handler": "some_string", "source": "docs"}}, id="handler-not-callable"),
        ],
    )
    def test_init_invalid_output_structure(self, outputs_to_state):
        with pytest.raises(ValueError):
            Tool(
                name="irrelevant",
                description="irrelevant",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                function=get_weather_report,
                outputs_to_state=outputs_to_state,
            )

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
        with pytest.raises(
            ToolInvocationError,
            match=re.escape(
                "Failed to invoke Tool `weather` with parameters {}. Error: get_weather_report() missing 1 required positional argument: 'city'"
            ),
        ):
            tool.invoke()

    def test_to_dict(self):
        tool = Tool(
            name="weather",
            description="Get weather report",
            parameters=parameters,
            function=get_weather_report,
            outputs_to_state={"documents": {"handler": get_weather_report, "source": "docs"}},
        )

        assert tool.to_dict() == {
            "type": "haystack.tools.tool.Tool",
            "data": {
                "name": "weather",
                "description": "Get weather report",
                "parameters": parameters,
                "function": "test_tool.get_weather_report",
                "outputs_to_string": None,
                "inputs_from_state": None,
                "outputs_to_state": {"documents": {"source": "docs", "handler": "test_tool.get_weather_report"}},
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
                "outputs_to_state": {"documents": {"source": "docs", "handler": "test_tool.get_weather_report"}},
            },
        }

        tool = Tool.from_dict(tool_dict)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report
        assert tool.outputs_to_state["documents"]["source"] == "docs"
        assert tool.outputs_to_state["documents"]["handler"] == get_weather_report


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
