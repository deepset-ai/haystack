# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, Toolset, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


def calculate(a: int, b: int, operation: str) -> int:
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0


def translate_text(text: str, target_language: str) -> str:
    return f"Translated '{text}' to {target_language}"


def summarize_text(text: str, max_length: int) -> str:
    return text[:max_length]


def format_text(text: str, style: str) -> str:
    return f"Formatted text in {style} style: {text}"


weather_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

calculator_parameters = {
    "type": "object",
    "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"},
        "operation": {"type": "string", "enum": ["add", "multiply"]},
    },
    "required": ["a", "b", "operation"],
}

translator_parameters = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
    "required": ["text", "target_language"],
}

summarizer_parameters = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "max_length": {"type": "integer"}},
    "required": ["text", "max_length"],
}

formatter_parameters = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "style": {"type": "string"}},
    "required": ["text", "style"],
}

# Legacy name for backward compatibility with existing tests
parameters = weather_parameters


class TestToolSerdeUtils:
    def test_serialize_toolset(self):
        toolset = Toolset(
            tools=[
                Tool(
                    name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
                )
            ]
        )

        data = serialize_tools_or_toolset(toolset)
        assert data == toolset.to_dict()

    def test_serialize_tool(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        data = serialize_tools_or_toolset([tool])
        assert data == [tool.to_dict()]

    def test_deserialize_tools_inplace(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )

        data = {"tools": [tool.to_dict()]}
        deserialize_tools_or_toolset_inplace(data)
        assert data["tools"] == [tool]

        data = {"mytools": [tool.to_dict()]}
        deserialize_tools_or_toolset_inplace(data, key="mytools")
        assert data["mytools"] == [tool]

        data = {"no_tools": 123}
        deserialize_tools_or_toolset_inplace(data)
        assert data == {"no_tools": 123}

    def test_deserialize_tools_inplace_failures(self):
        data = {"key": "value"}
        deserialize_tools_or_toolset_inplace(data)
        assert data == {"key": "value"}

        data = {"tools": None}
        deserialize_tools_or_toolset_inplace(data)
        assert data == {"tools": None}

        data = {"tools": "not a list"}
        with pytest.raises(TypeError):
            deserialize_tools_or_toolset_inplace(data)

        data = {"tools": ["not a dictionary"]}
        with pytest.raises(TypeError):
            deserialize_tools_or_toolset_inplace(data)

        # not a subclass of Tool
        data = {"tools": [{"type": "haystack.dataclasses.ChatMessage", "data": {"irrelevant": "irrelevant"}}]}
        with pytest.raises(TypeError):
            deserialize_tools_or_toolset_inplace(data)

    def test_deserialize_toolset_inplace(self):
        tool = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )
        toolset = Toolset(tools=[tool])

        data = {"tools": toolset.to_dict()}

        deserialize_tools_or_toolset_inplace(data)

        assert data["tools"] == toolset
        assert isinstance(data["tools"], Toolset)
        assert data["tools"][0] == tool

    def test_deserialize_toolset_inplace_failures(self):
        data = {"tools": {"key": "value"}}
        with pytest.raises(TypeError):
            deserialize_tools_or_toolset_inplace(data)

        data = {"tools": {"type": "haystack.tools.Tool", "data": "some_data"}}
        with pytest.raises(TypeError):
            deserialize_tools_or_toolset_inplace(data)

    def test_serialize_list_of_toolsets(self):
        """Test serialization of a list of Toolset instances."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=parameters, function=get_weather_report
        )

        toolset1 = Toolset([tool1])
        toolset2 = Toolset([tool2])

        data = serialize_tools_or_toolset([toolset1, toolset2])

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == toolset1.to_dict()
        assert data[1] == toolset2.to_dict()
        assert data[0]["type"] == "haystack.tools.toolset.Toolset"
        assert data[1]["type"] == "haystack.tools.toolset.Toolset"

    def test_deserialize_list_of_toolsets_inplace(self):
        """Test deserialization of a list of Toolset instances."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=parameters, function=get_weather_report
        )

        toolset1 = Toolset([tool1])
        toolset2 = Toolset([tool2])

        data = {"tools": [toolset1.to_dict(), toolset2.to_dict()]}
        deserialize_tools_or_toolset_inplace(data)

        assert isinstance(data["tools"], list)
        assert len(data["tools"]) == 2
        assert all(isinstance(ts, Toolset) for ts in data["tools"])
        assert data["tools"][0][0].name == "weather"
        assert data["tools"][1][0].name == "calculator"

    def test_serialize_mixed_list_tools_and_toolsets(self):
        """Test serialization of a mixed list of Tool and Toolset instances."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=weather_parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=calculator_parameters, function=calculate
        )

        toolset = Toolset([tool2])

        data = serialize_tools_or_toolset([tool1, toolset])

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == tool1.to_dict()
        assert data[0]["type"] == "haystack.tools.tool.Tool"
        assert data[0]["data"]["parameters"] == weather_parameters
        assert data[1] == toolset.to_dict()
        assert data[1]["type"] == "haystack.tools.toolset.Toolset"
        assert data[1]["data"]["tools"][0]["data"]["parameters"] == calculator_parameters

    def test_serialize_mixed_list_multiple_tools_and_toolsets(self):
        """Test serialization of a mixed list with multiple Tools and a Toolset containing multiple tools."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=weather_parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=calculator_parameters, function=calculate
        )
        tool3 = Tool(
            name="translator", description="Translate text", parameters=translator_parameters, function=translate_text
        )
        tool4 = Tool(
            name="summarizer", description="Summarize text", parameters=summarizer_parameters, function=summarize_text
        )
        tool5 = Tool(name="formatter", description="Format text", parameters=formatter_parameters, function=format_text)

        toolset = Toolset([tool4, tool5])

        data = serialize_tools_or_toolset([tool1, tool2, toolset, tool3])

        assert isinstance(data, list)
        assert len(data) == 4

        # Verify Tool 1 (weather)
        assert data[0] == tool1.to_dict()
        assert data[0]["type"] == "haystack.tools.tool.Tool"
        assert data[0]["data"]["name"] == "weather"
        assert data[0]["data"]["parameters"] == weather_parameters

        # Verify Tool 2 (calculator)
        assert data[1] == tool2.to_dict()
        assert data[1]["type"] == "haystack.tools.tool.Tool"
        assert data[1]["data"]["name"] == "calculator"
        assert data[1]["data"]["parameters"] == calculator_parameters

        # Verify Toolset (with summarizer and formatter)
        assert data[2] == toolset.to_dict()
        assert data[2]["type"] == "haystack.tools.toolset.Toolset"
        assert len(data[2]["data"]["tools"]) == 2
        assert data[2]["data"]["tools"][0]["data"]["name"] == "summarizer"
        assert data[2]["data"]["tools"][0]["data"]["parameters"] == summarizer_parameters
        assert data[2]["data"]["tools"][1]["data"]["name"] == "formatter"
        assert data[2]["data"]["tools"][1]["data"]["parameters"] == formatter_parameters

        # Verify Tool 3 (translator)
        assert data[3] == tool3.to_dict()
        assert data[3]["type"] == "haystack.tools.tool.Tool"
        assert data[3]["data"]["name"] == "translator"
        assert data[3]["data"]["parameters"] == translator_parameters

    def test_deserialize_mixed_list_tools_and_toolsets_inplace(self):
        """Test deserialization of a mixed list of Tool and Toolset instances."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=weather_parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=calculator_parameters, function=calculate
        )

        toolset = Toolset([tool2])

        data = {"tools": [tool1.to_dict(), toolset.to_dict()]}
        deserialize_tools_or_toolset_inplace(data)

        assert isinstance(data["tools"], list)
        assert len(data["tools"]) == 2

        # Verify Tool (weather)
        assert isinstance(data["tools"][0], Tool)
        assert data["tools"][0].name == "weather"
        assert data["tools"][0].parameters == weather_parameters
        assert data["tools"][0].function("Paris") == "Weather report for Paris: 20°C, sunny"

        # Verify Toolset with calculator tool
        assert isinstance(data["tools"][1], Toolset)
        assert len(data["tools"][1]) == 1
        assert data["tools"][1][0].name == "calculator"
        assert data["tools"][1][0].parameters == calculator_parameters
        assert data["tools"][1][0].function(10, 5, "add") == 15
        assert data["tools"][1][0].function(10, 5, "multiply") == 50

    def test_deserialize_mixed_list_multiple_tools_and_toolsets_inplace(self):
        """Test deserialization of a mixed list with multiple Tools and a Toolset containing multiple tools."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=weather_parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=calculator_parameters, function=calculate
        )
        tool3 = Tool(
            name="translator", description="Translate text", parameters=translator_parameters, function=translate_text
        )
        tool4 = Tool(
            name="summarizer", description="Summarize text", parameters=summarizer_parameters, function=summarize_text
        )
        tool5 = Tool(name="formatter", description="Format text", parameters=formatter_parameters, function=format_text)

        toolset = Toolset([tool4, tool5])

        data = {"tools": [tool1.to_dict(), tool2.to_dict(), toolset.to_dict(), tool3.to_dict()]}
        deserialize_tools_or_toolset_inplace(data)

        assert isinstance(data["tools"], list)
        assert len(data["tools"]) == 4

        # Verify Tool 1 (weather)
        assert isinstance(data["tools"][0], Tool)
        assert data["tools"][0].name == "weather"
        assert data["tools"][0].parameters == weather_parameters
        assert data["tools"][0].function("Berlin") == "Weather report for Berlin: 20°C, sunny"

        # Verify Tool 2 (calculator)
        assert isinstance(data["tools"][1], Tool)
        assert data["tools"][1].name == "calculator"
        assert data["tools"][1].parameters == calculator_parameters
        assert data["tools"][1].function(5, 3, "add") == 8
        assert data["tools"][1].function(5, 3, "multiply") == 15

        # Verify Toolset (with summarizer and formatter)
        assert isinstance(data["tools"][2], Toolset)
        assert len(data["tools"][2]) == 2
        assert data["tools"][2][0].name == "summarizer"
        assert data["tools"][2][0].parameters == summarizer_parameters
        assert data["tools"][2][0].function("Hello World", 5) == "Hello"
        assert data["tools"][2][1].name == "formatter"
        assert data["tools"][2][1].parameters == formatter_parameters
        assert data["tools"][2][1].function("test", "bold") == "Formatted text in bold style: test"

        # Verify Tool 3 (translator)
        assert isinstance(data["tools"][3], Tool)
        assert data["tools"][3].name == "translator"
        assert data["tools"][3].parameters == translator_parameters
        assert data["tools"][3].function("Hello", "Spanish") == "Translated 'Hello' to Spanish"

    def test_serialize_none_returns_none(self):
        """Test that serializing None returns None."""
        data = serialize_tools_or_toolset(None)
        assert data is None

    def test_serialize_empty_list_of_toolsets(self):
        """Test that serializing an empty list of Toolsets returns an empty list."""
        data = serialize_tools_or_toolset([])
        assert data == []
