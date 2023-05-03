from unittest import mock

import pytest
from haystack.agents.base import ToolsManager, Tool


@pytest.fixture
def tools_manager():
    tools = [
        Tool(name="ToolA", pipeline_or_node=mock.Mock(), description="Tool A Description"),
        Tool(name="ToolB", pipeline_or_node=mock.Mock(), description="Tool B Description"),
    ]
    return ToolsManager(tools=tools)


@pytest.mark.unit
def test_add_tool(tools_manager):
    new_tool = Tool(name="ToolC", pipeline_or_node=mock.Mock(), description="Tool C Description")
    tools_manager.add_tool(new_tool)
    assert "ToolC" in tools_manager.tools
    assert tools_manager.tools["ToolC"] == new_tool


@pytest.mark.unit
def test_get_tool_names(tools_manager):
    assert tools_manager.get_tool_names() == "ToolA, ToolB"


@pytest.mark.unit
def test_get_tools(tools_manager):
    tools = tools_manager.get_tools()
    assert len(tools) == 2
    assert tools[0].name == "ToolA"
    assert tools[1].name == "ToolB"


@pytest.mark.unit
def test_get_tool_names_with_descriptions(tools_manager):
    expected_output = "ToolA: Tool A Description\n" "ToolB: Tool B Description"
    assert tools_manager.get_tool_names_with_descriptions() == expected_output


@pytest.mark.unit
def test_extract_tool_name_and_tool_input(tools_manager):
    examples = [
        "need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy McKinnon born",
        "need to find out what city he was born.\n\nTool: Search\n\nTool Input: Where was Jeremy McKinnon born",
        "need to find out what city he was born. Tool: Search Tool Input: Where was Jeremy McKinnon born",
    ]
    for example in examples:
        tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name == "Search" and tool_input == "Where was Jeremy McKinnon born"

    negative_examples = [
        "need to find out what city he was born.",
        "Tool: Search",
        "Tool Input: Where was Jeremy McKinnon born",
        "need to find out what city he was born. Tool: Search",
        "Tool Input: Where was Jeremy McKinnon born",
    ]
    for example in negative_examples:
        tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name is None and tool_input is None
