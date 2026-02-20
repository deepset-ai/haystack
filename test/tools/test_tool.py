# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from haystack.dataclasses import TextContent
from haystack.tools import Tool, _check_duplicate_tool_names
from haystack.tools.errors import ToolInvocationError
from haystack.tools.tool import _deserialize_outputs_to_string, _serialize_outputs_to_string


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


def format_string(text: str) -> str:
    return f"Formatted: {text}"


def outputs_to_result_handler(result):
    return [TextContent(text=result["text"])]


parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


async def async_get_weather(city: str) -> str:
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
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_init_invalid_parameters(self):
        params = {"type": "invalid", "properties": {"city": {"type": "string"}}}
        with pytest.raises(ValueError):
            Tool(name="irrelevant", description="irrelevant", parameters=params, function=get_weather_report)

    def test_init_async_function_raises_error(self):
        with pytest.raises(ValueError, match="Async functions are not supported as tools"):
            Tool(name="weather", description="Get weather report", parameters=parameters, function=async_get_weather)

    @pytest.mark.parametrize(
        "outputs_to_state",
        [
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

    def test_init_invalid_output_structure_config_not_dict(self):
        with pytest.raises(TypeError):
            Tool(
                name="irrelevant",
                description="irrelevant",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                function=get_weather_report,
                outputs_to_state={"documents": ["some_value"]},
            )

    @pytest.mark.parametrize(
        "outputs_to_string",
        [
            pytest.param({"source": get_weather_report}, id="source-not-a-string"),
            pytest.param({"handler": "some_string"}, id="handler-not-callable"),
            pytest.param({"raw_result": "not-a-bool"}, id="raw_result-not-a-bool"),
            pytest.param({"documents": {"source": get_weather_report}}, id="multi-value-source-not-a-string"),
            pytest.param({"documents": {"handler": "some_string"}}, id="multi-value-handler-not-callable"),
            pytest.param(
                {"documents": {"source": "docs", "raw_result": True}}, id="multi-value-raw_result-not-supported"
            ),
        ],
    )
    def test_init_invalid_outputs_to_string_structure(self, outputs_to_string):
        with pytest.raises(ValueError):
            Tool(
                name="irrelevant",
                description="irrelevant",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                function=get_weather_report,
                outputs_to_string=outputs_to_string,
            )

    def test_init_invalid_outputs_to_string_structure_config_not_dict(self):
        with pytest.raises(TypeError):
            Tool(
                name="irrelevant",
                description="irrelevant",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                function=get_weather_report,
                outputs_to_string={"documents": ["some_value"]},
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
                "Failed to invoke Tool `weather` with parameters {}. Error: get_weather_report() missing 1 required "
                "positional argument: 'city'"
            ),
        ):
            tool.invoke()

    def test_to_dict(self):
        tool = Tool(
            name="weather",
            description="Get weather report",
            parameters=parameters,
            function=get_weather_report,
            outputs_to_string={"handler": format_string},
            inputs_from_state={"location": "city"},
            outputs_to_state={"documents": {"handler": get_weather_report, "source": "docs"}},
        )

        assert tool.to_dict() == {
            "type": "haystack.tools.tool.Tool",
            "data": {
                "name": "weather",
                "description": "Get weather report",
                "parameters": parameters,
                "function": "test_tool.get_weather_report",
                "outputs_to_string": {"handler": "test_tool.format_string"},
                "inputs_from_state": {"location": "city"},
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
                "outputs_to_string": {"handler": "test_tool.format_string"},
                "inputs_from_state": {"location": "city"},
                "outputs_to_state": {"documents": {"source": "docs", "handler": "test_tool.get_weather_report"}},
            },
        }

        tool = Tool.from_dict(tool_dict)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report
        assert tool.outputs_to_string == {"handler": format_string}
        assert tool.inputs_from_state == {"location": "city"}
        assert tool.outputs_to_state == {"documents": {"source": "docs", "handler": get_weather_report}}

    def test_serialize_outputs_to_string(self):
        config = {"handler": format_string, "source": "result", "raw_result": False}
        serialized = _serialize_outputs_to_string(config)
        assert serialized == {"handler": "test_tool.format_string", "source": "result", "raw_result": False}

        config = {"handler": format_string}
        serialized = _serialize_outputs_to_string(config)
        assert serialized == {"handler": "test_tool.format_string"}

        config = {"handler": outputs_to_result_handler, "raw_result": True}
        serialized = _serialize_outputs_to_string(config)
        assert serialized == {"handler": "test_tool.outputs_to_result_handler", "raw_result": True}

        config = {
            "report": {"source": "report", "handler": format_string},
            "temp": {"source": "temperature", "handler": format_string},
        }
        serialized = _serialize_outputs_to_string(config)
        assert serialized == {
            "report": {"source": "report", "handler": "test_tool.format_string"},
            "temp": {"source": "temperature", "handler": "test_tool.format_string"},
        }

    def test_deserialize_outputs_to_string(self):
        serialized = {"handler": "test_tool.format_string", "source": "result", "raw_result": False}
        deserialized = _deserialize_outputs_to_string(serialized)
        assert deserialized == {"handler": format_string, "source": "result", "raw_result": False}

        serialized = {"handler": "test_tool.format_string"}
        deserialized = _deserialize_outputs_to_string(serialized)
        assert deserialized == {"handler": format_string}

        serialized = {"handler": "test_tool.outputs_to_result_handler", "raw_result": True}
        deserialized = _deserialize_outputs_to_string(serialized)
        assert deserialized == {"handler": outputs_to_result_handler, "raw_result": True}

        serialized = {
            "report": {"source": "report", "handler": "test_tool.format_string"},
            "temp": {"source": "temperature", "handler": "test_tool.format_string"},
        }
        deserialized = _deserialize_outputs_to_string(serialized)
        assert deserialized == {
            "report": {"source": "report", "handler": format_string},
            "temp": {"source": "temperature", "handler": format_string},
        }

    def test_inputs_from_state_validation_with_invalid_parameter(self):
        """Test that inputs_from_state is validated against the parameters schema"""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "inputs_from_state maps 'state_key' to unknown parameter 'nonexistent'. Valid parameters are: {'city'}."
            ),
        ):
            Tool(
                name="weather",
                description="Get weather report",
                parameters=parameters,
                function=get_weather_report,
                inputs_from_state={"state_key": "nonexistent"},
            )

    def test_inputs_from_state_validation_with_non_string_value(self):
        """Test that inputs_from_state values must be strings"""
        with pytest.raises(TypeError, match=re.escape("inputs_from_state values must be str, not dict")):
            Tool(
                name="weather",
                description="Get weather report",
                parameters=parameters,
                function=get_weather_report,
                inputs_from_state={"state_key": {"source": "city"}},
            )

    def test_inputs_from_state_validation_with_valid_parameter(self):
        """Test that inputs_from_state works with valid parameter names"""
        tool = Tool(
            name="weather",
            description="Get weather report",
            parameters=parameters,
            function=get_weather_report,
            inputs_from_state={"location": "city"},
        )
        assert tool.inputs_from_state == {"location": "city"}

    def test_outputs_to_state_no_validation_when_get_valid_outputs_returns_none(self):
        """Test that outputs_to_state is not validated when _get_valid_outputs returns None"""
        # This should not raise an error even though "nonexistent" is not a valid output
        # because the base Tool class returns None from _get_valid_outputs()
        tool = Tool(
            name="weather",
            description="Get weather report",
            parameters=parameters,
            function=get_weather_report,
            outputs_to_state={"result": {"source": "nonexistent"}},
        )
        assert tool.outputs_to_state == {"result": {"source": "nonexistent"}}

    def test_outputs_to_state_validation_when_subclass_provides_valid_outputs(self):
        """Test that outputs_to_state is validated when subclass overrides _get_valid_outputs"""

        class ToolWithOutputs(Tool):
            def _get_valid_outputs(self):
                return {"report", "temperature"}

        # Valid output should work
        tool = ToolWithOutputs(
            name="weather",
            description="Get weather report",
            parameters=parameters,
            function=get_weather_report,
            outputs_to_state={"result": {"source": "report"}},
        )
        assert tool.outputs_to_state == {"result": {"source": "report"}}

        # Invalid output should raise an error
        with pytest.raises(
            ValueError,
            match=re.escape("outputs_to_state: 'weather' maps state key 'result' to unknown output 'nonexistent'"),
        ):
            ToolWithOutputs(
                name="weather",
                description="Get weather report",
                parameters=parameters,
                function=get_weather_report,
                outputs_to_state={"result": {"source": "nonexistent"}},
            )


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
