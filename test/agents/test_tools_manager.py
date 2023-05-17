import unittest
from unittest import mock

import pytest

from haystack import Pipeline, Answer, Document
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


@pytest.mark.unit
def test_invalid_tool_creation():
    with pytest.raises(ValueError, match="Invalid"):
        Tool(name="Tool-A", pipeline_or_node=mock.Mock(), description="Tool A Description")


@pytest.mark.unit
def test_tool_invocation():
    # by default for pipelines as tools we look for results key in the output
    p = Pipeline()
    tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description")
    with unittest.mock.patch("haystack.pipelines.Pipeline.run", return_value={"results": "mock"}):
        assert tool.run("input") == "mock"

    # now fail if results key is not present
    with unittest.mock.patch("haystack.pipelines.Pipeline.run", return_value={"no_results": "mock"}):
        with pytest.raises(ValueError, match="Tool ToolA returned result"):
            assert tool.run("input")

    # now try tool with a correct output variable
    tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description", output_variable="no_results")
    with unittest.mock.patch("haystack.pipelines.Pipeline.run", return_value={"no_results": "mock_no_results"}):
        assert tool.run("input") == "mock_no_results"

    # try tool that internally returns an Answer object but we extract the string
    tool = Tool(name="ToolA", pipeline_or_node=p, description="Tool A Description")
    with unittest.mock.patch("haystack.pipelines.Pipeline.run", return_value=[Answer("mocked_answer")]):
        assert tool.run("input") == "mocked_answer"

    # same but for the document
    with unittest.mock.patch("haystack.pipelines.Pipeline.run", return_value=[Document("mocked_document")]):
        assert tool.run("input") == "mocked_document"
