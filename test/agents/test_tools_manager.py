import unittest
from typing import Optional, Union, List, Dict, Any
from unittest import mock

import pytest

from haystack import Pipeline, Answer, Document, BaseComponent, MultiLabel
from haystack.agents.base import ToolsManager, Tool


@pytest.fixture
def tools_manager():
    tools = [
        Tool(name="ToolA", pipeline_or_node=mock.Mock(), description="Tool A Description"),
        Tool(name="ToolB", pipeline_or_node=mock.Mock(), description="Tool B Description"),
    ]
    return ToolsManager(tools=tools)


@pytest.mark.unit
def test_using_callable_as_tool():
    # test that we can also pass a callable as a tool
    tool_input = "Haystack"
    tool = Tool(name="ToolA", pipeline_or_node=lambda x: x + x, description="Tool A Description")
    assert tool.run(tool_input) == tool_input + tool_input


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
        'need to find out what city he was born. Tool: Search Tool Input: "Where was Jeremy McKinnon born"',
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


@pytest.mark.unit
def test_extract_tool_name_and_tool_multi_line_input(tools_manager):
    # new pattern being supported but with backward compatibility for the old
    text = (
        "We need to find out the following information:\n"
        "1. What city was Jeremy McKinnon born in?\n"
        "2. What's the capital of Germany?\n"
        "Tool: Search\n"
        "Tool Input: Where was Jeremy\n McKinnon born\n and where did he grow up?"
    )

    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text)
    assert tool_name == "Search" and tool_input == "Where was Jeremy\n McKinnon born\n and where did he grow up?"

    # tool input is empty
    text2 = (
        "We need to find out the following information:\n"
        "1. What city was Jeremy McKinnon born in?\n"
        "2. What's the capital of Germany?\n"
        "Tool: Search\n"
        "Tool Input:"
    )
    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text2)
    assert tool_name == "Search" and tool_input == ""

    # Case where the tool name and tool input are provided with extra whitespaces
    text3 = "   Tool:   Search   \n   Tool Input:   What is the tallest building in the world?   "
    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text3)
    assert tool_name.strip() == "Search" and tool_input.strip() == "What is the tallest building in the world?"

    # Case where the tool name is provided but the tool input line is not provided at all
    # Tool input is not optional, so this should return None for both tool name and tool input
    text4 = (
        "We need to find out the following information:\n"
        "1. Who is the current president of the United States?\n"
        "Tool: Search\n"
    )
    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text4)
    assert tool_name is None and tool_input is None

    # Case where neither the tool name nor the tool input is provided
    text5 = "We need to find out the following information:\n 1. What is the population of India?"
    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text5)
    assert tool_name is None and tool_input is None

    # Case where the tool name and tool input are provided with extra whitespaces and new lines
    text6 = "   Tool:   Search   \n   Tool Input:   \nWhat is the tallest \nbuilding in the world?   "
    tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(text6)
    assert tool_name.strip() == "Search" and tool_input.strip() == "What is the tallest \nbuilding in the world?"


@pytest.mark.unit
def test_extract_tool_name_and_empty_tool_input(tools_manager):
    examples = [
        "need to find out what city he was born.\nTool: Search\nTool Input:",
        "need to find out what city he was born.\nTool: Search\nTool Input:  ",
    ]
    for example in examples:
        tool_name, tool_input = tools_manager.extract_tool_name_and_tool_input(example)
        assert tool_name == "Search" and tool_input == ""


@pytest.mark.unit
def test_node_as_tool():
    # test that a component can be used as a tool
    class ToolComponent(BaseComponent):
        outgoing_edges = 1

        def run_batch(
            self,
            queries: Optional[Union[str, List[str]]] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
            documents: Optional[Union[List[Document], List[List[Document]]]] = None,
            meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            params: Optional[dict] = None,
            debug: Optional[bool] = None,
        ):
            pass

        def run(self, **kwargs):
            return "mocked_output"

    tool = Tool(name="ToolA", pipeline_or_node=ToolComponent(), description="Tool A Description")
    assert tool.run("input") == "mocked_output"


@pytest.mark.unit
def test_tools_manager_exception():
    # tests exception raising in tools manager
    class ToolComponent(BaseComponent):
        outgoing_edges = 1

        def run_batch(
            self,
            queries: Optional[Union[str, List[str]]] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
            documents: Optional[Union[List[Document], List[List[Document]]]] = None,
            meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            params: Optional[dict] = None,
            debug: Optional[bool] = None,
        ):
            pass

        def run(self, **kwargs):
            raise Exception("mocked_exception")

    fake_llm_response = "need to find out what city he was born.\nTool: Search\nTool Input: Where was Jeremy born"
    tool = Tool(name="Search", pipeline_or_node=ToolComponent(), description="Search")
    tools_manager = ToolsManager(tools=[tool])

    with pytest.raises(Exception):
        tools_manager.run_tool(llm_response=fake_llm_response)
