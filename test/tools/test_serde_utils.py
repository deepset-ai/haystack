# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, Toolset, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset


def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"


parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


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
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=parameters, function=get_weather_report
        )

        toolset = Toolset([tool2])

        data = serialize_tools_or_toolset([tool1, toolset])

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == tool1.to_dict()
        assert data[1] == toolset.to_dict()

    def test_deserialize_mixed_list_tools_and_toolsets_inplace(self):
        """Test deserialization of a mixed list of Tool and Toolset instances."""
        tool1 = Tool(
            name="weather", description="Get weather report", parameters=parameters, function=get_weather_report
        )
        tool2 = Tool(
            name="calculator", description="Calculate numbers", parameters=parameters, function=get_weather_report
        )

        toolset = Toolset([tool2])

        data = {"tools": [tool1.to_dict(), toolset.to_dict()]}
        deserialize_tools_or_toolset_inplace(data)

        assert isinstance(data["tools"], list)
        assert len(data["tools"]) == 2
        assert isinstance(data["tools"][0], Tool)
        assert isinstance(data["tools"][1], Toolset)
        assert data["tools"][0].name == "weather"
        assert data["tools"][1][0].name == "calculator"

    def test_serialize_none_returns_none(self):
        """Test that serializing None returns None."""
        data = serialize_tools_or_toolset(None)
        assert data is None

    def test_serialize_empty_list_of_toolsets(self):
        """Test that serializing an empty list of Toolsets returns an empty list."""
        data = serialize_tools_or_toolset([])
        assert data == []
