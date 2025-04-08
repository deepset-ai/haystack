# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, deserialize_tools_or_toolset_inplace, Toolset, serialize_tools_or_toolset


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
